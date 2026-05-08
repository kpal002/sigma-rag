"""
tests/test_pipeline.py
----------------------
Integration tests for SigmaRAGPipeline using the echo backend (no API key).

Tests cover:
  - Answerable queries return has_evidence=True with a non-empty answer
  - Unanswerable queries at strict threshold return has_evidence=False
  - compare_with_topk() returns both keys
  - context_used is populated for evidenced responses
  - Per-call n_sigma override works
"""

from __future__ import annotations

import pytest

from sigma_rag import SigmaIndex, SigmaRAGPipeline
from sigma_rag.types import RAGResponse


class TestPipelineEcho:
    """Pipeline tests using the echo (no-LLM) backend."""

    @pytest.fixture(scope="class")
    def pipeline(self, physics_index: SigmaIndex) -> SigmaRAGPipeline:
        """A SigmaRAGPipeline backed by the echo LLM (offline)."""
        return SigmaRAGPipeline(
            physics_index,
            n_sigma=1.0,  # permissive threshold for answerable queries
            max_results=5,
            llm="echo",
        )

    def test_query_returns_rag_response(self, pipeline: SigmaRAGPipeline) -> None:
        """query() should return a RAGResponse object."""
        response = pipeline.query("gravitational waves LIGO detection")
        assert isinstance(response, RAGResponse)

    def test_response_has_answer_string(self, pipeline: SigmaRAGPipeline) -> None:
        """response.answer should be a non-empty string."""
        response = pipeline.query("gravitational waves")
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_response_has_model_field(self, pipeline: SigmaRAGPipeline) -> None:
        """response.model should reflect the configured model."""
        response = pipeline.query("LIGO interferometer")
        assert response.model == "echo"

    def test_response_has_retrieval(self, pipeline: SigmaRAGPipeline) -> None:
        """response.retrieval should be a RetrievalResult."""
        from sigma_rag.types import RetrievalResult

        response = pipeline.query("gravitational waves")
        assert isinstance(response.retrieval, RetrievalResult)

    def test_evidenced_response_has_context(self, pipeline: SigmaRAGPipeline) -> None:
        """When has_evidence=True, context_used should be non-empty."""
        response = pipeline.query("matched filter signal noise")
        if response.has_evidence:
            assert len(response.context_used) > 0

    def test_unanswerable_at_strict_threshold(self, physics_index: SigmaIndex) -> None:
        """
        At very strict threshold (5σ), a cooking query should produce
        has_evidence=False and the suppression message.
        """
        strict_pipeline = SigmaRAGPipeline(
            physics_index,
            n_sigma=5.0,
            llm="echo",
        )
        response = strict_pipeline.query("best pasta carbonara recipe authentic Italian")
        assert response.has_evidence is False
        assert "No significant evidence" in response.answer or "σ-RAG" in response.answer

    def test_per_call_n_sigma_override(self, pipeline: SigmaRAGPipeline) -> None:
        """
        Passing n_sigma to query() should override the pipeline default
        for that call without mutating the pipeline.
        """
        original_n_sigma = pipeline.n_sigma
        response = pipeline.query("pasta recipe cooking", n_sigma=5.0)
        # The pipeline's n_sigma should be unchanged
        assert pipeline.n_sigma == original_n_sigma
        # At 5σ, the cooking query should produce no evidence
        assert response.has_evidence is False

    def test_compare_with_topk_returns_both_keys(self, pipeline: SigmaRAGPipeline) -> None:
        """compare_with_topk() should return dict with 'sigma_rag' and 'top_k'."""
        comparison = pipeline.compare_with_topk("LIGO gravitational waves", k=3)
        assert "sigma_rag" in comparison
        assert "top_k" in comparison

    def test_compare_topk_top_k_always_has_evidence(self, pipeline: SigmaRAGPipeline) -> None:
        """The top_k response should always report has_evidence=True."""
        comparison = pipeline.compare_with_topk("xyzzy nonsense query", k=3)
        assert comparison["top_k"].has_evidence is True

    def test_no_llm_call_when_no_evidence(self, physics_index: SigmaIndex) -> None:
        """
        When has_evidence=False, context_used should be empty string
        (LLM was never called).
        """
        strict = SigmaRAGPipeline(physics_index, n_sigma=5.0, llm="echo")
        response = strict.query("carbonara pancetta guanciale pepper cheese")
        if not response.has_evidence:
            assert response.context_used == ""


class TestPipelineUncalibrated:
    """Edge-case tests for pipeline behaviour with bad state."""

    def test_query_on_uncalibrated_index_raises(self) -> None:
        """pipeline.query() on an uncalibrated index should raise RuntimeError."""
        from sigma_rag.embedder import HashEmbedder

        index = SigmaIndex(embedder=HashEmbedder(), noise_n_pairs=50)
        index.add_documents(["Some text."])
        pipeline = SigmaRAGPipeline(index, llm="echo")
        with pytest.raises(RuntimeError):
            pipeline.query("test")
