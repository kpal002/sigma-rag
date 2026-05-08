"""
tests/test_noise_floor.py
-------------------------
Unit tests for NoiseFloor — the core statistical component of σ-RAG.

Tests cover:
  - fit() populates mu_, sigma_, n_pairs correctly
  - threshold() = mu_ + n * sigma_
  - z_score(), p_value() are monotonically sensible
  - is_significant() matches manual threshold comparison
  - false_alarm_rate() at n=0 ≈ 0.5, large n → 0
  - cross_doc_only sampling produces valid statistics
"""

from __future__ import annotations

import numpy as np
import pytest

from sigma_rag.noise_floor import NoiseFloor
from sigma_rag.embedder import HashEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n: int = 30, dim: int = 128, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """
    Create n random L2-normalised embeddings spread across 3 fake documents.

    Returns:
        embeddings: (n, dim) float32 array
        doc_ids:    list of 'doc_0', 'doc_1', 'doc_2' labels
    """
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    embeddings = vecs / norms
    doc_ids = [f"doc_{i % 3}" for i in range(n)]
    return embeddings, doc_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoiseFloorFit:
    """Test that NoiseFloor.fit() correctly populates internal state."""

    def test_fit_sets_parameters(self) -> None:
        """After fit(), mu_ and sigma_ should be finite floats."""
        nf = NoiseFloor()
        embeddings, doc_ids = _make_embeddings(30)
        nf.fit(embeddings, doc_ids=doc_ids, n_pairs=200, seed=0)
        assert np.isfinite(nf.mu_)
        assert np.isfinite(nf.sigma_)
        assert nf.sigma_ > 0

    def test_fit_records_n_pairs(self) -> None:
        """fit() should record the actual number of pairs sampled."""
        nf = NoiseFloor()
        embeddings, doc_ids = _make_embeddings(20)
        nf.fit(embeddings, doc_ids=doc_ids, n_pairs=100, seed=0)
        assert nf.stats is not None
        assert nf.stats.n_pairs > 0

    def test_fit_raises_before_fit(self) -> None:
        """Calling threshold() before fit() should raise RuntimeError."""
        nf = NoiseFloor()
        with pytest.raises(RuntimeError, match="not fitted"):
            nf.threshold(2.0)

    def test_fit_is_deterministic_with_seed(self) -> None:
        """Same seed should produce identical mu_ and sigma_."""
        embeddings, doc_ids = _make_embeddings(40)
        nf1 = NoiseFloor()
        nf2 = NoiseFloor()
        nf1.fit(embeddings, doc_ids=doc_ids, n_pairs=300, seed=99)
        nf2.fit(embeddings, doc_ids=doc_ids, n_pairs=300, seed=99)
        assert nf1.mu_ == nf2.mu_
        assert nf1.sigma_ == nf2.sigma_

    def test_cross_doc_only_sampling(self) -> None:
        """cross_doc_only=True should still produce a valid fit."""
        nf = NoiseFloor(cross_doc_only=True)
        embeddings, doc_ids = _make_embeddings(40)
        nf.fit(embeddings, doc_ids=doc_ids, n_pairs=300, seed=0)
        assert np.isfinite(nf.mu_)
        assert nf.sigma_ > 0


class TestNoiseFloorThreshold:
    """Test threshold, z-score, and p-value computations."""

    @pytest.fixture(autouse=True)
    def fitted_nf(self) -> None:
        """A fitted NoiseFloor shared across threshold tests."""
        embeddings, doc_ids = _make_embeddings(50)
        self.nf = NoiseFloor()
        self.nf.fit(embeddings, doc_ids=doc_ids, n_pairs=500, seed=0)

    def test_threshold_formula(self) -> None:
        """threshold(n) == mu_ + n * sigma_."""
        for n in [1.0, 2.0, 3.0]:
            expected = self.nf.mu_ + n * self.nf.sigma_
            assert abs(self.nf.threshold(n) - expected) < 1e-7

    def test_threshold_increases_with_n(self) -> None:
        """Higher n_sigma → higher threshold."""
        assert self.nf.threshold(3.0) > self.nf.threshold(2.0) > self.nf.threshold(1.0)

    def test_z_score_at_mean_is_zero(self) -> None:
        """z_score of the noise mean itself should be ≈ 0."""
        z = self.nf.z_score(self.nf.mu_)
        assert abs(z) < 1e-6

    def test_z_score_increases_with_similarity(self) -> None:
        """Higher similarity → higher z-score."""
        z_low = self.nf.z_score(self.nf.mu_ + 0.5 * self.nf.sigma_)
        z_high = self.nf.z_score(self.nf.mu_ + 2.0 * self.nf.sigma_)
        assert z_high > z_low

    def test_p_value_range(self) -> None:
        """p-value must be in [0, 1]."""
        for sim in [self.nf.mu_ - 2 * self.nf.sigma_,
                    self.nf.mu_,
                    self.nf.mu_ + 2 * self.nf.sigma_,
                    self.nf.mu_ + 5 * self.nf.sigma_]:
            p = self.nf.p_value(sim)
            assert 0.0 <= p <= 1.0

    def test_p_value_decreases_with_similarity(self) -> None:
        """Higher similarity → lower p-value (more significant)."""
        p_low_sim = self.nf.p_value(self.nf.mu_)
        p_high_sim = self.nf.p_value(self.nf.mu_ + 3 * self.nf.sigma_)
        assert p_low_sim > p_high_sim

    def test_is_significant_above_threshold(self) -> None:
        """A similarity above the threshold should be significant."""
        above = self.nf.mu_ + 3.0 * self.nf.sigma_
        assert self.nf.is_significant(above, n_sigma=2.0) is True

    def test_is_significant_below_threshold(self) -> None:
        """A similarity at the noise mean should not be significant."""
        assert self.nf.is_significant(self.nf.mu_, n_sigma=2.0) is False

    def test_false_alarm_rate_at_zero_sigma(self) -> None:
        """At n=0, FAR ≈ 0.5 (50% of noise pairs exceed the mean)."""
        far = self.nf.false_alarm_rate(0.0)
        assert 0.4 < far < 0.6

    def test_false_alarm_rate_decreases_with_n(self) -> None:
        """FAR should decrease monotonically with n_sigma."""
        far_1 = self.nf.false_alarm_rate(1.0)
        far_2 = self.nf.false_alarm_rate(2.0)
        far_3 = self.nf.false_alarm_rate(3.0)
        assert far_1 > far_2 > far_3

    def test_summary_returns_string(self) -> None:
        """summary() should return a non-empty string."""
        s = self.nf.summary()
        assert isinstance(s, str)
        assert len(s) > 0
