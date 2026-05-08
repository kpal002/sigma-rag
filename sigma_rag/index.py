"""
sigma_rag/index.py
------------------
Document ingestion, chunking, and embedding index.

SigmaIndex is the entry point for building a σ-RAG corpus.  It:
  1. Accepts raw documents (strings or (text, metadata) pairs).
  2. Chunks them into overlapping windows.
  3. Embeds each chunk.
  4. Calibrates the NoiseFloor from the resulting embeddings.

After calling .add_documents() and .calibrate(), the index is ready
to be passed to SigmaRetriever or SigmaRAGPipeline.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Union

import numpy as np

from sigma_rag.embedder import Embedder, SentenceTransformerEmbedder, HashEmbedder
from sigma_rag.noise_floor import NoiseFloor
from sigma_rag.types import Chunk

logger = logging.getLogger(__name__)

# Type alias: a document is either a plain string or a (text, metadata) tuple
Document = Union[str, tuple[str, dict]]


class SigmaIndex:
    """
    Document index with an embedded noise floor for σ-RAG retrieval.

    Args:
        embedder:       Embedder instance.  Defaults to SentenceTransformerEmbedder.
        chunk_size:     Maximum number of characters per chunk.
        chunk_overlap:  Character overlap between consecutive chunks (for
                        context continuity across chunk boundaries).
        n_sigma:        Default significance threshold used by SigmaRetriever
                        when this index is passed to it.
        noise_n_pairs:  Number of random pairs used to calibrate the noise floor.

    Example:
        >>> index = SigmaIndex()
        >>> index.add_documents(["Document one ...", "Document two ..."])
        >>> index.calibrate()
        >>> print(index.noise_floor.summary())
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        n_sigma: float = 2.0,
        noise_n_pairs: int = 10_000,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )

        # Use SentenceTransformerEmbedder when available, fall back to HashEmbedder
        if embedder is not None:
            self.embedder: Embedder = embedder
        else:
            try:
                self.embedder = SentenceTransformerEmbedder()
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using HashEmbedder (lower quality, no download needed). "
                    "Install sentence-transformers for production use."
                )
                self.embedder = HashEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_sigma = n_sigma
        self.noise_n_pairs = noise_n_pairs

        # Storage
        self._chunks: list[Chunk] = []
        self._embeddings_matrix: np.ndarray | None = None  # (n_chunks, d) float32
        self.noise_floor: NoiseFloor = NoiseFloor(cross_doc_only=True)
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        doc_ids: list[str] | None = None,
    ) -> "SigmaIndex":
        """
        Chunk and embed a list of documents, adding them to the index.

        Can be called multiple times to incrementally add documents.
        Call :meth:`calibrate` after all documents are added.

        Args:
            documents: List of raw strings, or list of (text, metadata) tuples.
            doc_ids:   Optional list of document identifiers.  If None,
                       auto-generated as 'doc_0', 'doc_1', ...

        Returns:
            Self, for chaining.
        """
        if doc_ids is None:
            offset = len(self._get_unique_doc_ids())
            doc_ids = [f"doc_{offset + i}" for i in range(len(documents))]

        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) must match "
                f"documents length ({len(documents)})."
            )

        new_chunks: list[Chunk] = []

        for doc_id, raw in zip(doc_ids, documents):
            # Unpack (text, metadata) or plain string
            if isinstance(raw, tuple):
                text, metadata = raw
            else:
                text, metadata = raw, {}

            chunks_text = self._chunk_text(text)
            for idx, chunk_text in enumerate(chunks_text):
                new_chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_idx=idx,
                        embedding=np.zeros(1, dtype=np.float32),  # placeholder
                        metadata={**metadata, "doc_id": doc_id},
                    )
                )

        if not new_chunks:
            logger.warning("No chunks produced from the provided documents.")
            return self

        # Embed all new chunks in one batched call
        logger.info("Embedding %d new chunks ...", len(new_chunks))
        texts = [c.text for c in new_chunks]
        embeddings = self.embedder.embed_batch(texts)  # (n_new, d)

        for chunk, emb in zip(new_chunks, embeddings):
            chunk.embedding = emb

        self._chunks.extend(new_chunks)
        self._rebuild_matrix()
        self._calibrated = False  # force re-calibration after new docs

        logger.info(
            "Index now contains %d chunks from %d unique documents.",
            len(self._chunks),
            len(self._get_unique_doc_ids()),
        )
        return self

    def add_document(
        self, text: str, doc_id: str | None = None, metadata: dict | None = None
    ) -> "SigmaIndex":
        """
        Convenience method to add a single document.

        Args:
            text:     Document text.
            doc_id:   Optional document identifier.
            metadata: Optional metadata dictionary.

        Returns:
            Self, for chaining.
        """
        doc: Document = (text, metadata or {}) if metadata else text
        ids = [doc_id] if doc_id else None
        return self.add_documents([doc], doc_ids=ids)

    # ------------------------------------------------------------------
    # Noise floor calibration
    # ------------------------------------------------------------------

    def calibrate(self, n_pairs: int | None = None, seed: int = 42) -> "SigmaIndex":
        """
        Fit the NoiseFloor on the current corpus embeddings.

        Must be called before the index is used for retrieval.  If you
        add more documents later, call calibrate() again.

        Args:
            n_pairs: Override the default noise_n_pairs for this call.
            seed:    RNG seed for reproducibility.

        Returns:
            Self, for chaining.

        Raises:
            RuntimeError: If no documents have been added yet.
        """
        if self._embeddings_matrix is None or len(self._chunks) == 0:
            raise RuntimeError(
                "No documents in the index.  Call add_documents() first."
            )

        doc_ids = [c.doc_id for c in self._chunks]
        n = n_pairs or self.noise_n_pairs

        logger.info(
            "Calibrating noise floor with %d pairs over %d chunks ...",
            n,
            len(self._chunks),
        )
        self.noise_floor.fit(
            self._embeddings_matrix,
            doc_ids=doc_ids,
            n_pairs=n,
            seed=seed,
        )
        self._calibrated = True
        logger.info(self.noise_floor.summary())
        return self

    # ------------------------------------------------------------------
    # Query helpers (used by SigmaRetriever)
    # ------------------------------------------------------------------

    def query_embeddings(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
            query: Query text.

        Returns:
            1-D float32 array of shape (d,), L2-normalised.
        """
        return self.embedder.embed(query)

    def cosine_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between the query and all indexed chunks.

        Because all embeddings are L2-normalised, cosine similarity is
        equivalent to the dot product — computable in one matrix multiply.

        Args:
            query_embedding: 1-D float32 array of shape (d,).

        Returns:
            1-D float32 array of shape (n_chunks,) with similarities in [-1, 1].
        """
        if self._embeddings_matrix is None:
            raise RuntimeError("Index is empty.  Call add_documents() first.")
        # (n_chunks, d) @ (d,) → (n_chunks,)
        return self._embeddings_matrix @ query_embedding

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> list[Chunk]:
        """All indexed chunks (read-only view)."""
        return list(self._chunks)

    @property
    def n_chunks(self) -> int:
        """Total number of indexed chunks."""
        return len(self._chunks)

    @property
    def calibrated(self) -> bool:
        """True if the noise floor has been fitted on the current corpus."""
        return self._calibrated

    def check_ready(self) -> None:
        """Raise RuntimeError if the index is not ready for retrieval."""
        if not self._chunks:
            raise RuntimeError("Index is empty. Call add_documents() first.")
        if not self._calibrated:
            raise RuntimeError(
                "Noise floor not calibrated. Call calibrate() after adding documents."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks of at most chunk_size characters.

        Args:
            text: Input text.

        Returns:
            List of chunk strings.  Always returns at least one chunk.
        """
        text = text.strip()
        if not text:
            return []

        # Fast path: fits in one chunk
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def _rebuild_matrix(self) -> None:
        """Stack all chunk embeddings into a single (n_chunks, d) matrix."""
        if not self._chunks:
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.stack(
            [c.embedding for c in self._chunks], axis=0
        ).astype(np.float32)

    def _get_unique_doc_ids(self) -> list[str]:
        """Return deduplicated list of doc_ids in insertion order."""
        seen: dict[str, None] = {}
        for c in self._chunks:
            seen[c.doc_id] = None
        return list(seen.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index (chunks, embeddings, noise floor) to disk.

        Args:
            path: Directory path.  Created if it does not exist.

        Example:
            >>> index.save("my_index")
            >>> loaded = SigmaIndex.load("my_index", embedder=embedder)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Embeddings matrix
        if self._embeddings_matrix is not None:
            np.save(path / "embeddings.npy", self._embeddings_matrix)

        # Noise floor state
        if self._calibrated and self.noise_floor.stats is not None:
            nf_state = {
                "mu_": self.noise_floor.mu_,
                "sigma_": self.noise_floor.sigma_,
                "cross_doc_only": self.noise_floor.cross_doc_only,
                "stats": {
                    "mu": self.noise_floor.stats.mu,
                    "sigma": self.noise_floor.stats.sigma,
                    "n_pairs": self.noise_floor.stats.n_pairs,
                    "ks_statistic": self.noise_floor.stats.ks_statistic,
                    "ks_p_value": self.noise_floor.stats.ks_p_value,
                },
            }
            (path / "noise_floor.json").write_text(json.dumps(nf_state))

        # Chunks (text + metadata, no embeddings — stored in matrix)
        chunk_data = [
            {
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        (path / "chunks.json").write_text(json.dumps(chunk_data))

        # Config
        config = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "n_sigma": self.n_sigma,
            "noise_n_pairs": self.noise_n_pairs,
            "calibrated": self._calibrated,
        }
        (path / "config.json").write_text(json.dumps(config))

        logger.info("SigmaIndex saved to %s (%d chunks)", path, len(self._chunks))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "SigmaIndex":
        """Load an index saved with :meth:`save`.

        Args:
            path:    Directory previously passed to :meth:`save`.
            embedder: Embedder to attach.  Must match the one used when
                      building the index (same embedding dim).  Defaults
                      to SentenceTransformerEmbedder / HashEmbedder fallback.

        Returns:
            A fully populated SigmaIndex ready for retrieval.
        """
        path = Path(path)
        config = json.loads((path / "config.json").read_text())

        index = cls.__new__(cls)
        index.chunk_size = config["chunk_size"]
        index.chunk_overlap = config["chunk_overlap"]
        index.n_sigma = config["n_sigma"]
        index.noise_n_pairs = config["noise_n_pairs"]
        index._calibrated = config["calibrated"]
        index._chunks = []
        index._embeddings_matrix = None

        if embedder is not None:
            index.embedder = embedder
        else:
            try:
                index.embedder = SentenceTransformerEmbedder()
            except ImportError:
                index.embedder = HashEmbedder()

        # Restore chunks
        chunk_data = json.loads((path / "chunks.json").read_text())
        emb_matrix: np.ndarray | None = None
        if (path / "embeddings.npy").exists():
            emb_matrix = np.load(path / "embeddings.npy")

        from sigma_rag.types import Chunk
        for i, cd in enumerate(chunk_data):
            emb = emb_matrix[i] if emb_matrix is not None else np.zeros(1, dtype=np.float32)
            index._chunks.append(
                Chunk(
                    text=cd["text"],
                    doc_id=cd["doc_id"],
                    chunk_idx=cd["chunk_idx"],
                    embedding=emb,
                    metadata=cd["metadata"],
                )
            )
        index._embeddings_matrix = emb_matrix

        # Restore noise floor
        index.noise_floor = NoiseFloor()
        nf_path = path / "noise_floor.json"
        if nf_path.exists() and index._calibrated:
            nf_state = json.loads(nf_path.read_text())
            index.noise_floor.mu_ = nf_state["mu_"]
            index.noise_floor.sigma_ = nf_state["sigma_"]
            index.noise_floor.cross_doc_only = nf_state["cross_doc_only"]
            index.noise_floor._fitted = True
            from sigma_rag.noise_floor import NoiseFloorStats
            index.noise_floor.stats = NoiseFloorStats(**nf_state["stats"])

        logger.info("SigmaIndex loaded from %s (%d chunks)", path, len(index._chunks))
        return index

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SigmaIndex(n_chunks={self.n_chunks}, "
            f"calibrated={self._calibrated}, "
            f"n_sigma={self.n_sigma})"
        )
