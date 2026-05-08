"""
sigma_rag/retriever.py
----------------------
Significance-filtered retrieval — the heart of σ-RAG.

SigmaRetriever wraps a SigmaIndex and applies a significance threshold
before returning any chunks to the caller.  Chunks whose cosine
similarity with the query does not clear the background distribution (at the
configured sigma level) are returned in a separate `noise` list rather
than passed to the LLM — preventing background contamination of the context.

Standard top-k retrieval (for comparison) is also implemented as
TopKRetriever using the same interface.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from sigma_rag.index import SigmaIndex
from sigma_rag.types import RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Protocol (interface) — makes it easy to swap retrieval strategies
# ------------------------------------------------------------------


class Retriever(Protocol):
    """
    Protocol that all retriever implementations must satisfy.

    Any class implementing a ``retrieve(query, **kwargs) -> RetrievalResult``
    method is compatible with SigmaRAGPipeline.
    """

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """Return a RetrievalResult for the given query."""
        ...


# ------------------------------------------------------------------
# σ-RAG retriever — significance-filtered
# ------------------------------------------------------------------


class SigmaRetriever:
    """
    Retrieves chunks that clear the noise floor at a given sigma level.

    This is the core retrieval engine for σ-RAG.  It:
      1. Embeds the query.
      2. Computes cosine similarity against all indexed chunks.
      3. Partitions chunks into *significant* (above threshold) and
         *noise* (below threshold).
      4. Returns a RetrievalResult with both lists.

    Crucially, if *no* chunk clears the threshold, the result has
    ``has_evidence=False`` — allowing the pipeline to respond with
    "I don't have enough information" rather than fabricating an answer
    from background-level context.  This mirrors the particle physics
    convention of not reporting a discovery when the excess is consistent
    with background fluctuations.

    Args:
        index:          A calibrated SigmaIndex.
        n_sigma:        Significance threshold in standard deviations.
                        Overrides index.n_sigma if provided.
        max_results:    Maximum number of significant chunks to return
                        (after significance filtering, the top-scoring
                        chunks are returned up to this limit).
        min_results:    If fewer than min_results significant chunks are
                        found, the threshold is relaxed in 0.5σ steps
                        until min_results are found OR n_sigma drops
                        below min_sigma.  Set to 0 to disable.
        min_sigma:      Floor for adaptive threshold relaxation.

    Example:
        >>> index = SigmaIndex()
        >>> index.add_documents(docs)
        >>> index.calibrate()
        >>> retriever = SigmaRetriever(index, n_sigma=2.0)
        >>> result = retriever.retrieve("What is dark matter?")
        >>> if result.has_evidence:
        ...     for sc in result.significant:
        ...         print(sc.z_score, sc.chunk.text[:60])
    """

    def __init__(
        self,
        index: SigmaIndex,
        n_sigma: float | None = None,
        max_results: int = 5,
        min_results: int = 0,
        min_sigma: float = 0.5,
    ) -> None:
        self.index = index
        self.n_sigma = n_sigma if n_sigma is not None else index.n_sigma
        self.max_results = max_results
        self.min_results = min_results
        self.min_sigma = min_sigma

    def retrieve(self, query: str, n_sigma: float | None = None) -> RetrievalResult:
        """
        Run significance-filtered retrieval for a query string.

        Args:
            query:   The natural language query.
            n_sigma: Per-call override for the significance threshold.
                     If None, uses the instance-level n_sigma.

        Returns:
            RetrievalResult with significant and noise chunk lists.

        Raises:
            RuntimeError: If the index is not calibrated.
        """
        self.index.check_ready()

        effective_sigma = n_sigma if n_sigma is not None else self.n_sigma

        # ── Step 1: embed query ──────────────────────────────────────
        query_emb = self.index.query_embeddings(query)

        # ── Step 2: cosine similarities against all chunks ──────────
        sims = self.index.cosine_similarities(query_emb)  # (n_chunks,)

        # ── Step 3: compute per-chunk statistics ─────────────────────
        nf = self.index.noise_floor
        scored = self._score_chunks(sims, effective_sigma)

        # ── Step 4: adaptive threshold relaxation (optional) ─────────
        if self.min_results > 0:
            scored, effective_sigma = self._maybe_relax_threshold(scored, effective_sigma)

        # ── Step 5: partition into significant / noise ────────────────
        significant = [sc for sc in scored if sc.significant]
        noise = [sc for sc in scored if not sc.significant]

        # Sort significant by z_score descending, cap at max_results
        significant.sort(key=lambda sc: sc.z_score, reverse=True)
        significant = significant[: self.max_results]

        logger.debug(
            "retrieve() | query=%r | sigma=%.1f | thresh=%.4f | significant=%d | noise=%d",
            query[:50],
            effective_sigma,
            nf.threshold(effective_sigma),
            len(significant),
            len(noise),
        )

        return RetrievalResult(
            query=query,
            significant=significant,
            noise=noise,
            threshold=nf.threshold(effective_sigma),
            n_sigma=effective_sigma,
            noise_mu=nf.mu_,
            noise_sigma=nf.sigma_,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_chunks(self, sims: np.ndarray, n_sigma: float) -> list[ScoredChunk]:
        """Convert raw similarity array to a list of ScoredChunk objects."""
        nf = self.index.noise_floor
        threshold = nf.threshold(n_sigma)

        from sigma_rag import stats as _stats

        sims_f = sims.astype(np.float64)
        z_scores = (sims_f - nf.mu_) / nf.sigma_
        p_values = [_stats.sf(float(z)) for z in z_scores]

        scored: list[ScoredChunk] = []
        chunks = self.index._chunks  # access internal list directly; avoid copy
        for i, chunk in enumerate(chunks):
            sim_f = float(sims_f[i])
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    similarity=sim_f,
                    z_score=float(z_scores[i]),
                    p_value=float(p_values[i]),
                    significant=sim_f >= threshold,
                )
            )

        return scored

    def _maybe_relax_threshold(
        self,
        scored: list[ScoredChunk],
        current_sigma: float,
    ) -> tuple[list[ScoredChunk], float]:
        """Relax significance threshold in 0.5σ steps if fewer than
        min_results significant chunks are found.

        Z-scores and p-values don't change with sigma — only the significant
        flag is updated, avoiding redundant full rescoring.

        Returns:
            Tuple of (updated chunks, effective sigma used).
        """
        sig_count = sum(1 for sc in scored if sc.significant)
        if sig_count >= self.min_results:
            return scored, current_sigma

        nf = self.index.noise_floor
        trial_sigma = current_sigma - 0.5
        while trial_sigma >= self.min_sigma:
            new_threshold = nf.threshold(trial_sigma)
            new_sig_count = sum(1 for sc in scored if sc.similarity >= new_threshold)
            logger.debug(
                "Threshold relaxed to %.1fσ → %d significant chunks",
                trial_sigma,
                new_sig_count,
            )
            if new_sig_count >= self.min_results:
                for sc in scored:
                    sc.significant = sc.similarity >= new_threshold
                return scored, trial_sigma
            trial_sigma -= 0.5

        # Could not reach min_results even at min_sigma — return as-is
        return scored, current_sigma


# ------------------------------------------------------------------
# Standard top-k retriever (baseline for benchmarking)
# ------------------------------------------------------------------


class TopKRetriever:
    """
    Standard top-k retriever (no significance filtering).

    Returns the k most similar chunks regardless of whether they
    clear any significance threshold.  Used as the baseline comparison
    in benchmarks to demonstrate the advantage of σ-RAG.

    Implements the same Retriever protocol as SigmaRetriever so it
    can be swapped in/out of SigmaRAGPipeline.

    Args:
        index:   A calibrated SigmaIndex.  (Calibration needed to
                 populate z_score / p_value in the returned objects,
                 but no threshold is applied.)
        k:       Number of chunks to return.
    """

    def __init__(self, index: SigmaIndex, k: int = 5) -> None:
        self.index = index
        self.k = k

    def retrieve(self, query: str, **_kwargs) -> RetrievalResult:
        """
        Retrieve top-k chunks by cosine similarity (no threshold).

        Args:
            query: The natural language query.

        Returns:
            RetrievalResult where ALL returned chunks are marked
            significant=True (no filtering applied).
        """
        self.index.check_ready()

        query_emb = self.index.query_embeddings(query)
        sims = self.index.cosine_similarities(query_emb)
        nf = self.index.noise_floor

        # Score all chunks
        all_scored: list[ScoredChunk] = []
        for chunk, sim in zip(self.index.chunks, sims, strict=False):
            sim_f = float(sim)
            all_scored.append(
                ScoredChunk(
                    chunk=chunk,
                    similarity=sim_f,
                    z_score=nf.z_score(sim_f),
                    p_value=nf.p_value(sim_f),
                    significant=True,  # top-k: no filtering
                )
            )

        # Sort and take top-k
        all_scored.sort(key=lambda sc: sc.similarity, reverse=True)
        top_k = all_scored[: self.k]

        return RetrievalResult(
            query=query,
            significant=top_k,  # all returned chunks labelled significant
            noise=[],  # nothing discarded
            threshold=nf.threshold(self.index.n_sigma),
            n_sigma=self.index.n_sigma,
            noise_mu=nf.mu_,
            noise_sigma=nf.sigma_,
        )
