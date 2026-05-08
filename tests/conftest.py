"""
tests/conftest.py
-----------------
Shared pytest fixtures for the σ-RAG test suite.

All fixtures use HashEmbedder (pure numpy, no downloads) so the entire
test suite runs offline without any API keys or model downloads.
"""

from __future__ import annotations

import pytest

from sigma_rag import SigmaIndex
from sigma_rag.embedder import HashEmbedder

# ---------------------------------------------------------------------------
# Sample corpora
# ---------------------------------------------------------------------------

PHYSICS_DOCS = [
    (
        "Gravitational waves are ripples in spacetime caused by massive accelerating objects. "
        "They were predicted by Einstein's general theory of relativity in 1916 and first "
        "directly detected by LIGO in 2015. The waves carry energy away from the source.",
        {"domain": "gw_physics"},
    ),
    (
        "LIGO uses laser interferometry to detect gravitational waves. Two perpendicular arms "
        "each 4 km long bounce laser beams to measure changes in length smaller than 1/1000th "
        "the diameter of a proton. Seismic isolation and quantum squeezing reduce noise.",
        {"domain": "gw_instrument"},
    ),
    (
        "The matched filter is the optimal linear detector for a known signal waveform in "
        "Gaussian noise, per the Neyman-Pearson lemma. It maximises the signal-to-noise ratio "
        "by correlating the observed data stream with a template waveform bank.",
        {"domain": "signal_processing"},
    ),
    (
        "The Standard Model of particle physics describes three of the four fundamental forces "
        "and classifies all known elementary particles. It includes quarks, leptons, gauge bosons, "
        "and the Higgs boson discovered at CERN in 2012.",
        {"domain": "particle_physics"},
    ),
    (
        "Python is a high-level, general-purpose programming language emphasising readability. "
        "It supports multiple programming paradigms including procedural, object-oriented, and "
        "functional programming. NumPy provides efficient numerical array operations.",
        {"domain": "programming"},
    ),
    (
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even "
        "light, can escape beyond the event horizon. Hawking radiation is a predicted quantum effect "
        "whereby black holes slowly lose mass over astronomical timescales.",
        {"domain": "gw_physics"},
    ),
    (
        "Neutron stars are ultra-dense stellar remnants formed after supernova explosions. "
        "Binary neutron star mergers produce both gravitational waves and electromagnetic signals, "
        "an event known as a kilonova. GW170817 was the first multi-messenger detection.",
        {"domain": "gw_physics"},
    ),
    (
        "The Fourier transform decomposes a signal into its constituent frequency components. "
        "The Fast Fourier Transform algorithm computes this in O(n log n) operations, enabling "
        "real-time spectral analysis in gravitational wave data pipelines.",
        {"domain": "signal_processing"},
    ),
    (
        "Quantum mechanics governs physical phenomena at atomic and subatomic scales. "
        "The Schrodinger equation describes how the quantum state evolves in time. "
        "Quantum squeezing reduces shot noise below the standard quantum limit in LIGO.",
        {"domain": "particle_physics"},
    ),
    (
        "Machine learning models extract patterns from large datasets. Deep neural networks "
        "use layered nonlinear transformations to learn hierarchical feature representations. "
        "Transformer architectures dominate natural language processing tasks.",
        {"domain": "programming"},
    ),
]

COOKING_DOCS = [
    (
        "Pasta carbonara is a Roman dish made with eggs, hard cheese, cured pork, and black "
        "pepper. The creamy sauce is formed by emulsifying the egg yolks with pasta cooking water "
        "off the heat — no cream is used in the traditional recipe.",
        {"domain": "cooking"},
    ),
    (
        "Sourdough bread uses wild yeast and lactic acid bacteria for fermentation. The long "
        "cold-proof develops complex flavours and a chewy crumb structure. Hydration levels "
        "between 70-80% are typical for open crumb loaves.",
        {"domain": "cooking"},
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hash_embedder() -> HashEmbedder:
    """A deterministic HashEmbedder — no downloads required."""
    return HashEmbedder(embedding_dim=256, seed=42)


@pytest.fixture(scope="session")
def physics_index(hash_embedder: HashEmbedder) -> SigmaIndex:
    """
    A calibrated SigmaIndex over the physics corpus.

    Session-scoped so it's built once and reused across all tests.
    """
    index = SigmaIndex(
        embedder=hash_embedder,
        chunk_size=512,
        chunk_overlap=64,
        n_sigma=2.0,
        noise_n_pairs=500,  # small for speed in tests
    )
    index.add_documents(PHYSICS_DOCS)
    index.calibrate(seed=0)
    return index


@pytest.fixture(scope="session")
def mixed_index(hash_embedder: HashEmbedder) -> SigmaIndex:
    """
    A calibrated SigmaIndex over physics + cooking docs.

    Used for hallucination-prevention tests (unanswerable queries).
    """
    index = SigmaIndex(
        embedder=hash_embedder,
        chunk_size=512,
        chunk_overlap=64,
        n_sigma=2.0,
        noise_n_pairs=500,
    )
    index.add_documents(PHYSICS_DOCS + COOKING_DOCS)
    index.calibrate(seed=0)
    return index
