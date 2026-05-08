"""
sigma_rag/embedder.py
---------------------
Embedding abstraction layer.

Supports:
  - SentenceTransformerEmbedder  (local, no API key, recommended for dev)
  - OpenAIEmbedder               (text-embedding-3-small/large, for production)

All embedders return L2-normalised float32 vectors so that cosine
similarity reduces to a simple dot product — important for the noise
floor estimator, which uses batched einsum operations.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(abc.ABC):
    """
    Abstract base class for all embedding backends.

    Subclasses must implement :meth:`embed_batch`, which receives a list
    of strings and returns an (n, d) float32 array of L2-normalised vectors.
    """

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            Float32 array of shape (len(texts), embedding_dim),
            with each row L2-normalised to unit length.
        """
        ...

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Args:
            text: String to embed.

        Returns:
            1-D float32 array of shape (embedding_dim,), L2-normalised.
        """
        result: np.ndarray = self.embed_batch([text])[0]
        return result

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalise each row of a 2-D array in-place.

        Args:
            vectors: Array of shape (n, d).

        Returns:
            Same array with each row scaled to unit L2 norm.
            Rows with zero norm are left unchanged to avoid NaN.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for degenerate zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        normalized: np.ndarray = (vectors / norms).astype(np.float32)
        return normalized


class SentenceTransformerEmbedder(Embedder):
    """
    Local embedder backed by sentence-transformers.

    No API key required. Uses 'all-MiniLM-L6-v2' by default
    (384-dim, fast, good quality for English text).

    Args:
        model_name: Any model from https://huggingface.co/sentence-transformers.
                    Defaults to 'all-MiniLM-L6-v2'.
        batch_size: Number of texts to embed in one forward pass.
        device:     'cpu', 'cuda', or 'mps'. If None, auto-detected.

    Example:
        >>> embedder = SentenceTransformerEmbedder()
        >>> vecs = embedder.embed_batch(["hello world", "foo bar"])
        >>> vecs.shape
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        self.embedding_dim: int = dim if dim is not None else 384
        logger.info("Model loaded. Embedding dim: %d", self.embedding_dim)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts using the sentence-transformers model.

        Args:
            texts: List of strings.

        Returns:
            Float32 array of shape (len(texts), embedding_dim), L2-normalised.
        """
        # sentence-transformers normalise=True returns unit vectors already,
        # but we explicitly normalise for safety / consistency across backends.
        raw: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._l2_normalize(raw)


class OpenAIEmbedder(Embedder):
    """
    Embedder backed by OpenAI's text-embedding API.

    Requires OPENAI_API_KEY to be set in the environment.

    Args:
        model:      OpenAI embedding model name.
                    Defaults to 'text-embedding-3-small' (1536-dim, fast, cheap).
                    Use 'text-embedding-3-large' for higher quality (3072-dim).
        batch_size: Max texts per API call (OpenAI limit: 2048 inputs).

    Example:
        >>> embedder = OpenAIEmbedder()
        >>> vec = embedder.embed("What is dark matter?")
        >>> vec.shape
        (1536,)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 512,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIEmbedder. Install it with: pip install openai"
            ) from exc

        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI()  # reads OPENAI_API_KEY from env

        # Infer embedding dim from model name
        _dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.embedding_dim: int = _dim_map.get(model, 1536)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts via the OpenAI embedding API.

        Args:
            texts: List of strings (max 2048 per call).

        Returns:
            Float32 array of shape (len(texts), embedding_dim), L2-normalised.
        """
        all_embeddings: list[np.ndarray] = []

        # Batch to respect API input limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            batch_vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
            all_embeddings.append(batch_vecs)

        vectors = np.vstack(all_embeddings)
        return self._l2_normalize(vectors)


# ---------------------------------------------------------------------------
# Hash-based local embedder (zero dependencies, useful for testing)
# ---------------------------------------------------------------------------


class HashEmbedder(Embedder):
    """
    Deterministic bag-of-words embedder using only numpy.

    No model downloads, no API keys, no GPU needed. Suitable for
    unit tests, CI pipelines, and offline demos.

    Works by:
      1. Tokenising text into lowercased words.
      2. Hashing each word to a dimension index (mod embedding_dim).
      3. Accumulating TF-IDF-like weights per dimension.
      4. L2-normalising the result.

    Quality is far below sentence-transformers, but the noise floor
    estimation and significance logic work correctly with it.

    Args:
        embedding_dim: Dimensionality of the output vectors.
        seed:          RNG seed for hash offset (ensures different
                       vocabularies hash to different dims).

    Example:
        >>> embedder = HashEmbedder(dim=256)
        >>> v = embedder.embed("Higgs boson LHC ATLAS CMS")
        >>> v.shape
        (256,)
    """

    def __init__(self, embedding_dim: int = 256, seed: int = 0) -> None:
        self.embedding_dim = embedding_dim
        self._seed = seed

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts via hashed bag-of-words.

        Args:
            texts: List of strings.

        Returns:
            Float32 array of shape (len(texts), embedding_dim), L2-normalised.
        """
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z]+", text.lower())
            if not tokens:
                continue
            for tok in tokens:
                # Stable hash: SHA-256 of (seed prefix + token)
                h = hashlib.sha256(f"{self._seed}:{tok}".encode()).digest()
                # Use first 4 bytes as an unsigned int → dim index
                dim = int.from_bytes(h[:4], "big") % self.embedding_dim
                # Weight: 1 + log(freq) — simple TF approximation
                vectors[i, dim] += 1.0
        return self._l2_normalize(vectors)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_embedder(
    backend: str = "sentence_transformers",
    **kwargs,
) -> Embedder:
    """
    Convenience factory to instantiate an embedder by name.

    Args:
        backend: One of 'sentence_transformers' or 'openai'.
        **kwargs: Forwarded to the embedder constructor.

    Returns:
        An initialised Embedder instance.

    Raises:
        ValueError: If backend is not recognised.

    Example:
        >>> embedder = get_embedder("sentence_transformers", model_name="all-MiniLM-L6-v2")
    """
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(**kwargs)
    elif backend == "openai":
        return OpenAIEmbedder(**kwargs)
    elif backend == "hash":
        return HashEmbedder(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'sentence_transformers', 'openai', or 'hash'."
        )
