"""
sigma_rag/types.py
------------------
Core data classes for σ-RAG.

Analogy: just as particle physicists distinguish "background fluctuations"
from "significant excesses" (requiring e.g. 5σ for a discovery claim),
σ-RAG distinguishes retrieved chunks that clear a significance threshold
from those consistent with the background-only hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Chunk:
    """
    A single text chunk from a document, with its embedding.

    Attributes:
        text:       Raw text content of the chunk.
        doc_id:     Identifier of the source document.
        chunk_idx:  Position of this chunk within the source document.
        embedding:  L2-normalised embedding vector. Shape: (d,).
        metadata:   Arbitrary key-value metadata (filename, page, etc.).
    """

    text: str
    doc_id: str
    chunk_idx: int
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D105
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(doc={self.doc_id!r}, idx={self.chunk_idx}, text={preview!r}...)"


@dataclass
class ScoredChunk:
    """
    A chunk paired with its retrieval statistics.

    Attributes:
        chunk:       The underlying Chunk object.
        similarity:  Raw cosine similarity to the query. Range [-1, 1].
        z_score:     (similarity - background_mu) / background_sigma  — how many
                     standard deviations above the background this hit is.
                     Analogous to signal significance in particle physics
                     (e.g. the 5σ threshold used for Higgs discovery at the LHC).
        p_value:     One-tailed probability P(X ≥ similarity | H0), where
                     H0 is the null hypothesis that query and chunk are
                     unrelated. Small p_value → strong evidence of relevance.
        significant: True if z_score ≥ the configured sigma threshold.
    """

    chunk: Chunk
    similarity: float
    z_score: float
    p_value: float
    significant: bool

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ScoredChunk(z={self.z_score:.2f}, p={self.p_value:.4f}, "
            f"sig={self.significant}, text={self.chunk.text[:40]!r}...)"
        )


@dataclass
class RetrievalResult:
    """
    Full result of a σ-RAG retrieval call.

    Attributes:
        query:           The original query string.
        significant:     Chunks that cleared the significance threshold —
                         these are passed to the LLM as context.
        noise:           Chunks that were retrieved but did NOT clear the
                         threshold. Useful for debugging.
        threshold:       The similarity value used as the cutoff.
        n_sigma:         The sigma level used (e.g. 2.0 → ~2.3% false-alarm rate).
        noise_mu:        Estimated mean of the null distribution.
        noise_sigma:     Estimated std of the null distribution.
        has_evidence:    Convenience bool — True if at least one significant
                         chunk was found.
    """

    query: str
    significant: list[ScoredChunk]
    noise: list[ScoredChunk]
    threshold: float
    n_sigma: float
    noise_mu: float
    noise_sigma: float

    @property
    def has_evidence(self) -> bool:
        """True if at least one chunk cleared the significance threshold."""
        return len(self.significant) > 0

    @property
    def best(self) -> Optional[ScoredChunk]:
        """Highest-scoring significant chunk, or None."""
        return self.significant[0] if self.significant else None

    def summary(self) -> str:
        """Human-readable one-liner for logging / debugging."""
        return (
            f"RetrievalResult | query={self.query[:50]!r} | "
            f"significant={len(self.significant)} | "
            f"noise={len(self.noise)} | "
            f"threshold={self.threshold:.4f} ({self.n_sigma}σ)"
        )


@dataclass
class RAGResponse:
    """
    Final response from the full σ-RAG pipeline.

    Attributes:
        answer:         The generated answer string. If no significant
                        evidence was found, this is a canned "insufficient
                        evidence" message rather than a hallucination.
        retrieval:      The underlying RetrievalResult (for inspection /
                        logging).
        has_evidence:   Mirrors retrieval.has_evidence for convenience.
        model:          Name of the LLM used for generation.
        context_used:   The context string that was passed to the LLM.
                        Empty string if no evidence was found.
    """

    answer: str
    retrieval: RetrievalResult
    has_evidence: bool
    model: str
    context_used: str

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RAGResponse(has_evidence={self.has_evidence}, "
            f"answer={self.answer[:80]!r}...)"
        )
