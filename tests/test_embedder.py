"""
tests/test_embedder.py
----------------------
Unit tests for the embedding backends.

All tests use HashEmbedder (zero dependencies).
SentenceTransformer and OpenAI tests are marked integration and skipped offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigma_rag.embedder import Embedder, HashEmbedder, get_embedder


class TestHashEmbedder:
    """Tests for the pure-numpy HashEmbedder."""

    def test_embed_single_returns_1d(self) -> None:
        """embed() should return a 1-D array of the correct dimension."""
        embedder = HashEmbedder(embedding_dim=128)
        vec = embedder.embed("hello world")
        assert vec.ndim == 1
        assert vec.shape[0] == 128

    def test_embed_batch_shape(self) -> None:
        """embed_batch() should return (n, dim) float32 array."""
        embedder = HashEmbedder(embedding_dim=64)
        texts = ["foo", "bar", "baz"]
        vecs = embedder.embed_batch(texts)
        assert vecs.shape == (3, 64)
        assert vecs.dtype == np.float32

    def test_unit_norm(self) -> None:
        """All output vectors must have L2 norm ≈ 1.0."""
        embedder = HashEmbedder(embedding_dim=256)
        texts = ["gravitational waves", "quantum mechanics", "pasta carbonara"]
        vecs = embedder.embed_batch(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_determinism(self) -> None:
        """Same text + same seed → identical embedding."""
        e1 = HashEmbedder(embedding_dim=128, seed=7)
        e2 = HashEmbedder(embedding_dim=128, seed=7)
        text = "the matched filter is the optimal linear detector"
        np.testing.assert_array_equal(e1.embed(text), e2.embed(text))

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (almost always) produce different embeddings."""
        e1 = HashEmbedder(embedding_dim=256, seed=0)
        e2 = HashEmbedder(embedding_dim=256, seed=99)
        vec1 = e1.embed("LIGO interferometer")
        vec2 = e2.embed("LIGO interferometer")
        # Same words but different seed → different dimension assignments
        assert not np.allclose(vec1, vec2)

    def test_empty_text_does_not_crash(self) -> None:
        """Empty strings should return a valid (zero-like, then normed or left) vector."""
        embedder = HashEmbedder(embedding_dim=64)
        # Should not raise; result is undefined but must be a valid float32 array
        vec = embedder.embed("")
        assert vec.shape == (64,)
        assert np.all(np.isfinite(vec))

    def test_similar_texts_higher_cosine(self) -> None:
        """
        Two thematically similar texts should have higher cosine similarity
        than two unrelated texts, even with HashEmbedder.
        """
        embedder = HashEmbedder(embedding_dim=512)
        gw1 = embedder.embed("gravitational waves ripples spacetime LIGO detection")
        gw2 = embedder.embed("gravitational wave signal LIGO interferometer detector")
        cooking = embedder.embed("pasta carbonara eggs cheese pepper Roman recipe")
        sim_related = float(gw1 @ gw2)
        sim_unrelated = float(gw1 @ cooking)
        assert sim_related > sim_unrelated, (
            f"Expected related similarity ({sim_related:.3f}) > unrelated ({sim_unrelated:.3f})"
        )

    def test_get_embedder_factory_hash(self) -> None:
        """get_embedder('hash') should return a HashEmbedder instance."""
        embedder = get_embedder("hash", embedding_dim=32)
        assert isinstance(embedder, HashEmbedder)
        assert isinstance(embedder, Embedder)

    def test_get_embedder_invalid_backend(self) -> None:
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_embedder("nonexistent_backend")
