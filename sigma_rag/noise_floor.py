"""
sigma_rag/noise_floor.py
------------------------
Background-distribution estimator for cosine similarity in embedding space.

Physics analogy
---------------
In particle physics experiments (e.g. ATLAS or CMS at the LHC), before
claiming a new-particle discovery you must first characterise the
*background* — the distribution of events expected from known Standard
Model processes in the absence of any new signal.  This background is
estimated from control regions or sidebands in data, then extrapolated
into the signal region.  A discovery is only claimed when the observed
excess over background reaches a local significance of 5σ (local
p-value < 2.87 × 10⁻⁷), meaning a background-only fluctuation of that
size would occur by chance less than once in ~3.5 million trials.

Here we apply the same logic to embedding space:

  1. Sample many *random* cross-document pairs from the corpus.
     These represent the null (background-only) hypothesis H₀:
     "query and document are unrelated."

  2. Compute cosine similarity for every sampled pair.
     The resulting distribution is approximately Gaussian with mean μ
     and standard deviation σ — this is our background estimate.

  3. Set a significance threshold θ = μ + n_σ · σ.
     Any query-document similarity above θ is considered a significant
     excess over background (local p-value < Φ(-n_σ)).

The key insight: standard top-k retrieval always returns the top-k
chunks regardless of whether any of them are above background.
σ-RAG only returns chunks that clear the significance threshold —
preventing background-level context from contaminating the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sigma_rag import stats  # pure-numpy fallback; uses scipy when available

logger = logging.getLogger(__name__)


@dataclass
class NoiseFloorStats:
    """
    Summary statistics of the fitted null distribution.

    Attributes:
        mu:             Mean cosine similarity between random pairs.
        sigma:          Std dev of cosine similarity between random pairs.
        n_pairs:        Number of random pairs used to fit the distribution.
        ks_statistic:   Kolmogorov-Smirnov statistic for the Gaussian fit.
                        Low values (< 0.05) indicate a good Gaussian fit.
        ks_p_value:     p-value for the KS test.  High values are good.
    """

    mu: float
    sigma: float
    n_pairs: int
    ks_statistic: float
    ks_p_value: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"NoiseFloorStats(μ={self.mu:.4f}, σ={self.sigma:.4f}, "
            f"n_pairs={self.n_pairs}, KS p={self.ks_p_value:.3f})"
        )


class NoiseFloor:
    """
    Estimates the background distribution of cosine similarities.

    The background distribution characterises how similar two *unrelated*
    embeddings are — the analogue of the expected background yield in a
    particle physics signal search.  It is estimated by sampling random
    cross-document pairs from the corpus, which serve as a proxy for the
    background-only hypothesis H₀ ("query and document are unrelated").

    Once fitted, the object exposes:
      - :meth:`threshold`      — the similarity cutoff at n_σ above background
      - :meth:`z_score`        — convert similarity → signal significance (σ above background)
      - :meth:`p_value`        — local p-value under H₀
      - :meth:`is_significant` — Boolean gate at a given sigma level

    Args:
        cross_doc_only: If True, only sample pairs from different source
                        documents (avoids inflating noise floor with chunks
                        from the same document, which are naturally similar).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> embeddings = rng.standard_normal((200, 384)).astype(np.float32)
        >>> # L2-normalise (mimics real embedding output)
        >>> embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        >>> nf = NoiseFloor()
        >>> nf.fit(embeddings)
        >>> print(nf.stats)
        >>> print(nf.threshold(n_sigma=2.0))
    """

    def __init__(self, cross_doc_only: bool = True) -> None:
        self.cross_doc_only = cross_doc_only
        self._fitted = False
        self.mu_: float = 0.0
        self.sigma_: float = 1.0
        self.stats: NoiseFloorStats | None = None
        self._null_samples: np.ndarray | None = None  # kept for diagnostics

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str] | None = None,
        n_pairs: int = 10_000,
        seed: int = 42,
    ) -> NoiseFloor:
        """
        Estimate the null distribution from random document-pair similarities.

        Args:
            embeddings:  Float32 array of shape (n_chunks, d), L2-normalised.
            doc_ids:     Optional list of length n_chunks identifying which
                         source document each chunk belongs to.  Used for
                         cross_doc_only filtering.
            n_pairs:     Number of random pairs to sample.  More pairs give a
                         more stable estimate; 5000–20000 is a good range.
            seed:        RNG seed for reproducibility.

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If fewer than 10 embeddings are provided.
        """
        n = len(embeddings)
        if n < 10:
            raise ValueError(f"Need at least 10 embeddings to estimate noise floor, got {n}.")

        rng = np.random.default_rng(seed)

        if self.cross_doc_only and doc_ids is not None:
            sims = self._sample_cross_doc_pairs(embeddings, doc_ids, n_pairs, rng)
        else:
            sims = self._sample_random_pairs(embeddings, n_pairs, rng)

        # Fit Gaussian to the empirical distribution
        self.mu_ = float(np.mean(sims))
        self.sigma_ = float(np.std(sims, ddof=1))

        if self.sigma_ < 1e-8:
            logger.warning(
                "Noise floor σ is near zero (%.2e). All embeddings may be identical or nearly so.",
                self.sigma_,
            )
            self.sigma_ = 1e-6  # prevent division by zero downstream

        # Goodness-of-fit: KS test against fitted Gaussian
        ks_stat, ks_p = stats.ks_test(sims, self.mu_, self.sigma_)

        self.stats = NoiseFloorStats(
            mu=self.mu_,
            sigma=self.sigma_,
            n_pairs=len(sims),
            ks_statistic=float(ks_stat),
            ks_p_value=float(ks_p),
        )
        self._null_samples = sims
        self._fitted = True

        logger.info(
            "NoiseFloor fitted: μ=%.4f, σ=%.4f, n_pairs=%d, KS p=%.3f",
            self.mu_,
            self.sigma_,
            len(sims),
            ks_p,
        )

        if ks_p < 0.01:
            logger.warning(
                "KS test p=%.4f < 0.01: null distribution deviates from "
                "Gaussian.  Thresholds are approximate.  Consider increasing "
                "n_pairs or using a topic-diverse calibration corpus.",
                ks_p,
            )

        return self

    # ------------------------------------------------------------------
    # Core statistical operations
    # ------------------------------------------------------------------

    def threshold(self, n_sigma: float = 2.0) -> float:
        """
        Return the similarity threshold corresponding to n_sigma above noise.

        Analogous to the signal significance threshold in particle physics:
        3σ for "evidence," 5σ for "discovery" (the LHC standard).

        At n_sigma=2.0 the false-alarm probability is ≈ 2.3%.
        At n_sigma=3.0 it drops to ≈ 0.13%.
        At n_sigma=5.0 it drops to ≈ 2.9 × 10⁻⁷ (the LHC discovery bar).

        Args:
            n_sigma: Number of standard deviations above the noise mean.

        Returns:
            Similarity threshold (float).
        """
        self._check_fitted()
        return self.mu_ + n_sigma * self.sigma_

    def z_score(self, similarity: float) -> float:
        """
        Convert a raw cosine similarity to a σ-score (SNR above noise).

        Args:
            similarity: Cosine similarity value in [-1, 1].

        Returns:
            z-score: (similarity - μ) / σ
        """
        self._check_fitted()
        return (similarity - self.mu_) / self.sigma_

    def p_value(self, similarity: float) -> float:
        """
        One-tailed p-value: P(X ≥ similarity | H₀).

        A small p-value (e.g. < 0.05) means the observed similarity is
        unlikely under the null hypothesis that the pair is unrelated.

        Args:
            similarity: Cosine similarity value.

        Returns:
            p-value in [0, 1].
        """
        self._check_fitted()
        z = self.z_score(similarity)
        # One-tailed: probability of seeing similarity THIS high by chance
        return stats.sf(z)  # one-tailed: P(Z > z)

    def is_significant(self, similarity: float, n_sigma: float = 2.0) -> bool:
        """
        Return True if similarity clears the noise floor at n_sigma.

        Args:
            similarity: Cosine similarity to test.
            n_sigma:    Significance level in standard deviations.

        Returns:
            True if similarity >= threshold(n_sigma).
        """
        return similarity >= self.threshold(n_sigma)

    def false_alarm_rate(self, n_sigma: float) -> float:
        """
        Expected fraction of background (unrelated) pairs that would clear
        the threshold at n_sigma — the false-alarm rate under H₀.
        Analogous to the local p-value in a particle physics counting experiment.

        Args:
            n_sigma: Sigma level of the threshold.

        Returns:
            False alarm rate in [0, 1].
        """
        return stats.sf(n_sigma)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the fitted noise floor."""
        self._check_fitted()
        assert self.stats is not None
        far_2s = self.false_alarm_rate(2.0)
        far_3s = self.false_alarm_rate(3.0)
        return (
            f"Noise Floor Summary\n"
            f"  μ (noise mean)     : {self.mu_:.4f}\n"
            f"  σ (noise std)      : {self.sigma_:.4f}\n"
            f"  Threshold @ 2σ     : {self.threshold(2.0):.4f}  "
            f"(FAR ≈ {far_2s * 100:.2f}%)\n"
            f"  Threshold @ 3σ     : {self.threshold(3.0):.4f}  "
            f"(FAR ≈ {far_3s * 100:.3f}%)\n"
            f"  KS test p-value    : {self.stats.ks_p_value:.4f}  "
            f"({'OK' if self.stats.ks_p_value > 0.01 else 'WARNING: non-Gaussian'})\n"
            f"  Calibration pairs  : {self.stats.n_pairs}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_random_pairs(
        self,
        embeddings: np.ndarray,
        n_pairs: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample n_pairs random distinct-index pairs and compute cosine sims."""
        n = len(embeddings)
        max_possible = n * (n - 1) // 2
        effective = min(n_pairs, max_possible)
        if effective < n_pairs:
            logger.warning(
                "Corpus has only %d chunks (%d unique pairs). "
                "Requested %d noise pairs; using %d with replacement.",
                n,
                max_possible,
                n_pairs,
                effective,
            )

        sims: list[float] = []
        max_attempts = effective * 10 + 1000
        attempts = 0
        while len(sims) < effective and attempts < max_attempts:
            batch = min(effective - len(sims), 2000)
            idx_a = rng.integers(0, n, size=batch)
            idx_b = rng.integers(0, n, size=batch)
            mask = idx_a != idx_b
            idx_a, idx_b = idx_a[mask], idx_b[mask]
            if len(idx_a) > 0:
                batch_sims = np.einsum("ij,ij->i", embeddings[idx_a], embeddings[idx_b])
                sims.extend(batch_sims.tolist())
            attempts += batch

        if len(sims) < effective:
            logger.warning("Could only collect %d random pairs (wanted %d).", len(sims), effective)

        return np.array(sims[:effective], dtype=np.float32)

    def _sample_cross_doc_pairs(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str],
        n_pairs: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample pairs from different source documents only."""
        doc_id_array = np.array(doc_ids)
        n = len(embeddings)
        sims: list[float] = []
        max_attempts = n_pairs * 5  # guard against infinite loop on small corpora

        attempts = 0
        while len(sims) < n_pairs and attempts < max_attempts:
            batch = min(n_pairs - len(sims), 2000)
            idx_a = rng.integers(0, n, size=batch)
            idx_b = rng.integers(0, n, size=batch)
            # Keep only cross-document and distinct-index pairs
            cross_doc = (doc_id_array[idx_a] != doc_id_array[idx_b]) & (idx_a != idx_b)
            idx_a, idx_b = idx_a[cross_doc], idx_b[cross_doc]
            if len(idx_a) > 0:
                batch_sims = np.einsum("ij,ij->i", embeddings[idx_a], embeddings[idx_b])
                sims.extend(batch_sims.tolist())
            attempts += batch

        if len(sims) < n_pairs:
            logger.warning(
                "Only collected %d cross-doc pairs (wanted %d). "
                "Falling back to random pairs for remainder.",
                len(sims),
                n_pairs,
            )
            # Fill with random pairs for the remainder
            remainder = self._sample_random_pairs(embeddings, n_pairs - len(sims), rng)
            sims.extend(remainder.tolist())

        return np.array(sims[:n_pairs], dtype=np.float32)

    def _check_fitted(self) -> None:
        """Raise if fit() has not been called yet."""
        if not self._fitted:
            raise RuntimeError("NoiseFloor is not fitted yet. Call .fit(embeddings) first.")
