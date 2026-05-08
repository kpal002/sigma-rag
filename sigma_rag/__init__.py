"""
σ-RAG  (Sigma-RAG)
==================
Significance-threshold retrieval for RAG pipelines, inspired by
signal significance testing in particle physics.

Standard RAG always returns the top-k chunks regardless of whether
they are actually relevant — you may be injecting pure background into
the LLM's context.  σ-RAG instead estimates the background distribution
of your embedding space (the distribution of similarities between
unrelated document pairs) and only passes chunks that clear a
configurable significance threshold (default: 2σ above background).

Quick start
-----------
>>> from sigma_rag import SigmaIndex, SigmaRAGPipeline
>>> index = SigmaIndex()
>>> index.add_documents(["Doc text one ...", "Doc text two ..."])
>>> index.calibrate()          # fits the noise floor
>>>
>>> pipeline = SigmaRAGPipeline(index)
>>> response = pipeline.query("What is ...?")
>>> print(response.answer)
"""

from sigma_rag.embedder import Embedder, HashEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
from sigma_rag.index import SigmaIndex
from sigma_rag.noise_floor import NoiseFloor
from sigma_rag.pipeline import SigmaRAGPipeline
from sigma_rag.retriever import SigmaRetriever
from sigma_rag.types import Chunk, RAGResponse, RetrievalResult, ScoredChunk

__all__ = [
    "Chunk",
    "ScoredChunk",
    "RetrievalResult",
    "RAGResponse",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "NoiseFloor",
    "SigmaIndex",
    "SigmaRetriever",
    "SigmaRAGPipeline",
    "__version__",
]

__version__ = "0.1.0"
