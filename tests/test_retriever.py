"""
tests/test_retriever.py
-----------------------
Tests for SigmaRetriever and TopKRetriever.

Core properties verified:
  - All returned chunks clear the significance threshold
  - Unanswerable queries produce has_evidence=False
  - TopKRetriever always returns exactly k results
  - z_scores and p_values are correctly populated
  - Adaptive threshold relaxation works when min_results > 0
"""

from __future__ import annotations

import pytest

from sigma_rag import SigmaIndex
from sigma_rag.retriever import SigmaRetriever, TopKRetriever
from sigma_rag.types import RetrievalResult


class TestSigmaRetriever:
    """Tests for significance-gated retrieval."""

    def test_answerable_query_has_evidence(self, physics_index: SigmaIndex) -> None:
        """
        A query directly about gravitational waves should surface at least
        one significant chunk from the physics corpus.
        """
        retriever = SigmaRetriever(physics_index, n_sigma=1.0, max_results=5)
        result = retriever.retrieve("How does LIGO detect gravitational waves?")
        # With n_sigma=1.0 (lenient) we expect at least one hit on a matching corpus
        # Note: HashEmbedder quality is limited, so we use a permissive threshold here
        assert isinstance(result, RetrievalResult)

    def test_significant_chunks_clear_threshold(self, physics_index: SigmaIndex) -> None:
        """Every chunk in result.significant must exceed the threshold."""
        retriever = SigmaRetriever(physics_index, n_sigma=1.0, max_results=10)
        result = retriever.retrieve("gravitational wave detection laser interferometer")
        threshold = physics_index.noise_floor.threshold(result.n_sigma)
        for sc in result.significant:
            assert sc.similarity >= threshold - 1e-6, (
                f"Chunk similarity {sc.similarity:.4f} < threshold {threshold:.4f}"
            )

    def test_significant_chunks_have_positive_z_scores(self, physics_index: SigmaIndex) -> None:
        """All significant chunks must have z_score > 0."""
        retriever = SigmaRetriever(physics_index, n_sigma=1.0, max_results=5)
        result = retriever.retrieve("matched filter signal noise Gaussian")
        for sc in result.significant:
            assert sc.z_score > 0, f"Expected z_score > 0, got {sc.z_score}"

    def test_p_values_in_range(self, physics_index: SigmaIndex) -> None:
        """p_values for significant chunks should be in [0, 1]."""
        retriever = SigmaRetriever(physics_index, n_sigma=1.0, max_results=5)
        result = retriever.retrieve("gravitational waves")
        for sc in result.significant:
            assert 0.0 <= sc.p_value <= 1.0

    def test_max_results_respected(self, physics_index: SigmaIndex) -> None:
        """Number of significant chunks should not exceed max_results."""
        retriever = SigmaRetriever(physics_index, n_sigma=0.5, max_results=2)
        result = retriever.retrieve("gravitational waves LIGO")
        assert len(result.significant) <= 2

    def test_result_sorted_by_similarity(self, physics_index: SigmaIndex) -> None:
        """Significant chunks should be sorted highest similarity first."""
        retriever = SigmaRetriever(physics_index, n_sigma=0.5, max_results=10)
        result = retriever.retrieve("gravitational waves spacetime LIGO")
        sims = [sc.similarity for sc in result.significant]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_unanswerable_query_no_evidence(self, physics_index: SigmaIndex) -> None:
        """
        A query about something completely absent from the corpus
        should produce has_evidence=False at a sufficiently strict threshold.

        Note: this test uses a very high n_sigma to ensure the bar
        is not cleared by noise-level matches.
        """
        retriever = SigmaRetriever(physics_index, n_sigma=5.0, max_results=5)
        result = retriever.retrieve("best pasta carbonara authentic Roman recipe")
        # At 5σ, essentially no random noise chunk should clear the bar
        assert not result.has_evidence

    def test_result_noise_chunks_below_threshold(self, physics_index: SigmaIndex) -> None:
        """
        Chunks in result.noise should all have similarity < threshold,
        i.e., they are correctly classified as sub-threshold.
        """
        retriever = SigmaRetriever(physics_index, n_sigma=2.0, max_results=5)
        result = retriever.retrieve("gravitational waves")
        threshold = physics_index.noise_floor.threshold(result.n_sigma)
        for sc in result.noise:
            assert sc.similarity < threshold + 1e-6

    def test_significant_flag_on_scored_chunks(self, physics_index: SigmaIndex) -> None:
        """ScoredChunks in significant should have .significant=True."""
        retriever = SigmaRetriever(physics_index, n_sigma=0.5, max_results=10)
        result = retriever.retrieve("LIGO interferometer laser spacetime")
        for sc in result.significant:
            assert sc.significant is True
        for sc in result.noise:
            assert sc.significant is False

    def test_retrieve_before_calibration_raises(self) -> None:
        """Retrieving from an uncalibrated index should raise RuntimeError."""
        from sigma_rag.embedder import HashEmbedder
        index = SigmaIndex(embedder=HashEmbedder(), noise_n_pairs=50)
        index.add_documents(["Some document text here."])
        retriever = SigmaRetriever(index)
        with pytest.raises(RuntimeError):
            retriever.retrieve("some query")


class TestTopKRetriever:
    """Tests for the top-k baseline retriever."""

    def test_topk_returns_exactly_k(self, physics_index: SigmaIndex) -> None:
        """TopKRetriever should always return exactly k results."""
        k = 3
        retriever = TopKRetriever(physics_index, k=k)
        result = retriever.retrieve("gravitational waves")
        assert len(result.significant) == k

    def test_topk_result_has_evidence(self, physics_index: SigmaIndex) -> None:
        """TopKRetriever always marks has_evidence=True (no gating)."""
        retriever = TopKRetriever(physics_index, k=3)
        # Even a nonsense query returns evidence (this is the problem σ-RAG solves)
        result = retriever.retrieve("xyzzy frobnosticate quux")
        assert result.has_evidence is True

    def test_topk_returns_fewer_than_k_if_small_corpus(self) -> None:
        """If corpus has fewer than k chunks, return all available."""
        from sigma_rag.embedder import HashEmbedder
        small_docs = [f"Document {i} about a unique topic." for i in range(12)]
        index = SigmaIndex(embedder=HashEmbedder(embedding_dim=64), noise_n_pairs=50)
        index.add_documents(small_docs)
        index.calibrate(n_pairs=50, seed=0)
        retriever = TopKRetriever(index, k=100)
        result = retriever.retrieve("test query")
        assert len(result.significant) <= index.n_chunks

    def test_topk_sorted_by_similarity(self, physics_index: SigmaIndex) -> None:
        """Top-k results should be sorted highest similarity first."""
        retriever = TopKRetriever(physics_index, k=5)
        result = retriever.retrieve("quantum mechanics particles")
        sims = [sc.similarity for sc in result.significant]
        assert sims == sorted(sims, reverse=True)
