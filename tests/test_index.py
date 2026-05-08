"""
tests/test_index.py
-------------------
Unit tests for SigmaIndex — document ingestion, chunking, and calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigma_rag import SigmaIndex
from sigma_rag.embedder import HashEmbedder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_index() -> SigmaIndex:
    """A fresh, uncalibrated SigmaIndex with HashEmbedder."""
    return SigmaIndex(
        embedder=HashEmbedder(embedding_dim=128),
        chunk_size=200,
        chunk_overlap=32,
        noise_n_pairs=200,
    )


SAMPLE_DOCS = [
    "Gravitational waves carry energy away from massive accelerating systems.",
    "LIGO uses laser interferometry to detect spacetime strain.",
    "The matched filter maximises SNR for a known waveform in Gaussian noise.",
    "Black holes are regions where gravity prevents even light from escaping.",
    "Neutron stars are ultra-dense remnants of massive stellar explosions.",
    "The Fourier transform decomposes a signal into its frequency components.",
    "Quantum mechanics governs physical phenomena at subatomic scales.",
    "The Standard Model describes the fundamental particles and forces.",
    "General relativity predicts the curvature of spacetime by mass-energy.",
    "Signal-to-noise ratio measures the level of a signal against background noise.",
]


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------


class TestSigmaIndexIngestion:
    """Tests for add_documents() and add_document()."""

    def test_add_documents_increases_chunk_count(self, empty_index: SigmaIndex) -> None:
        """Adding documents should produce at least one chunk per document."""
        empty_index.add_documents(SAMPLE_DOCS)
        assert empty_index.n_chunks >= len(SAMPLE_DOCS)

    def test_add_document_singular(self, empty_index: SigmaIndex) -> None:
        """add_document() convenience method should work identically."""
        empty_index.add_document("A short test document.", doc_id="test_doc")
        assert empty_index.n_chunks >= 1

    def test_add_documents_with_metadata(self, empty_index: SigmaIndex) -> None:
        """Metadata attached to documents should be stored in chunks."""
        docs = [("Document with metadata.", {"source": "test", "year": 2024})]
        empty_index.add_documents(docs)
        chunk = empty_index.chunks[-1]
        assert chunk.metadata.get("source") == "test"

    def test_add_documents_with_doc_ids(self, empty_index: SigmaIndex) -> None:
        """Explicit doc_ids should be stored in chunks."""
        empty_index.add_documents(["Doc A", "Doc B"], doc_ids=["alpha", "beta"])
        doc_ids_in_index = {c.doc_id for c in empty_index.chunks}
        assert "alpha" in doc_ids_in_index
        assert "beta" in doc_ids_in_index

    def test_mismatched_doc_ids_raises(self, empty_index: SigmaIndex) -> None:
        """Passing wrong number of doc_ids should raise ValueError."""
        with pytest.raises(ValueError, match="doc_ids length"):
            empty_index.add_documents(["A", "B", "C"], doc_ids=["only_one"])

    def test_incremental_add(self, empty_index: SigmaIndex) -> None:
        """Two separate add_documents() calls should accumulate chunks."""
        empty_index.add_documents(["First batch."])
        count_after_first = empty_index.n_chunks
        empty_index.add_documents(["Second batch."])
        assert empty_index.n_chunks > count_after_first

    def test_calibrated_flag_resets_on_add(self, empty_index: SigmaIndex) -> None:
        """Adding new documents after calibration should mark index as uncalibrated."""
        empty_index.add_documents(SAMPLE_DOCS)
        empty_index.calibrate(n_pairs=100, seed=0)
        assert empty_index.calibrated
        empty_index.add_document("New document after calibration.")
        assert not empty_index.calibrated


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Tests for the _chunk_text internal method."""

    def test_short_text_is_single_chunk(self, empty_index: SigmaIndex) -> None:
        """Text shorter than chunk_size should produce exactly one chunk."""
        short = "Short."
        chunks = empty_index._chunk_text(short)
        assert len(chunks) == 1
        assert chunks[0] == short

    def test_long_text_produces_multiple_chunks(self) -> None:
        """Long text should be split into multiple overlapping chunks."""
        index = SigmaIndex(
            embedder=HashEmbedder(embedding_dim=64),
            chunk_size=50,
            chunk_overlap=10,
        )
        long_text = "A" * 200
        chunks = index._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunks_respect_max_size(self) -> None:
        """No chunk should exceed chunk_size characters."""
        index = SigmaIndex(
            embedder=HashEmbedder(embedding_dim=64),
            chunk_size=100,
            chunk_overlap=20,
        )
        long_text = "word " * 100  # 500 chars
        chunks = index._chunk_text(long_text)
        for c in chunks:
            assert len(c) <= 100

    def test_empty_text_returns_empty(self, empty_index: SigmaIndex) -> None:
        """Empty or whitespace-only text should return an empty list."""
        assert empty_index._chunk_text("") == []
        assert empty_index._chunk_text("   ") == []


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    """Tests for calibrate() and check_ready()."""

    def test_calibrate_marks_ready(self, empty_index: SigmaIndex) -> None:
        """After calibrate(), calibrated should be True."""
        empty_index.add_documents(SAMPLE_DOCS)
        empty_index.calibrate(n_pairs=100, seed=0)
        assert empty_index.calibrated

    def test_calibrate_before_add_raises(self, empty_index: SigmaIndex) -> None:
        """Calibrating on empty index should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            empty_index.calibrate()

    def test_check_ready_raises_if_not_calibrated(self, empty_index: SigmaIndex) -> None:
        """check_ready() on uncalibrated index should raise RuntimeError."""
        empty_index.add_documents(SAMPLE_DOCS)
        with pytest.raises(RuntimeError, match="calibrat"):
            empty_index.check_ready()

    def test_check_ready_passes_after_calibration(self, empty_index: SigmaIndex) -> None:
        """check_ready() should not raise on a properly calibrated index."""
        empty_index.add_documents(SAMPLE_DOCS)
        empty_index.calibrate(n_pairs=100, seed=0)
        empty_index.check_ready()  # Should not raise


# ---------------------------------------------------------------------------
# Embedding matrix tests
# ---------------------------------------------------------------------------


class TestEmbeddingMatrix:
    """Tests for the internal embedding matrix."""

    def test_embeddings_matrix_shape(self, empty_index: SigmaIndex) -> None:
        """_embeddings_matrix should be (n_chunks, embedding_dim) float32."""
        empty_index.add_documents(SAMPLE_DOCS)
        mat = empty_index._embeddings_matrix
        assert mat is not None
        assert mat.ndim == 2
        assert mat.shape[0] == empty_index.n_chunks
        assert mat.dtype == np.float32

    def test_cosine_similarities_shape(self, empty_index: SigmaIndex) -> None:
        """cosine_similarities() should return (n_chunks,) array."""
        empty_index.add_documents(SAMPLE_DOCS)
        empty_index.calibrate(n_pairs=100, seed=0)
        q_emb = empty_index.query_embeddings("gravitational waves")
        sims = empty_index.cosine_similarities(q_emb)
        assert sims.shape == (empty_index.n_chunks,)

    def test_cosine_similarities_in_range(self, empty_index: SigmaIndex) -> None:
        """Cosine similarities should be in [-1, 1] for unit vectors."""
        empty_index.add_documents(SAMPLE_DOCS)
        empty_index.calibrate(n_pairs=100, seed=0)
        q_emb = empty_index.query_embeddings("LIGO laser interferometer")
        sims = empty_index.cosine_similarities(q_emb)
        assert np.all(sims >= -1.01)
        assert np.all(sims <= 1.01)

    def test_repr(self, empty_index: SigmaIndex) -> None:
        """__repr__ should return a non-empty informative string."""
        r = repr(empty_index)
        assert "SigmaIndex" in r
        assert "n_chunks" in r
