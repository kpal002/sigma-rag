"""
validate.py
-----------
Offline smoke-test / validation script for sigma-rag.
Runs without pytest (no network required) and verifies all core modules.
"""

from __future__ import annotations

import sys
import traceback
import numpy as np

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
HEAD = "\033[1;94m"
RST  = "\033[0m"

results = []

def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"    {e}")
        results.append((name, False, str(e)))


# ── HashEmbedder ────────────────────────────────────────────────────────────
print(f"\n{HEAD}HashEmbedder{RST}")
from sigma_rag.embedder import HashEmbedder, get_embedder

def _embed_shape():
    e = HashEmbedder(embedding_dim=128)
    v = e.embed("hello world")
    assert v.shape == (128,), f"shape={v.shape}"

def _embed_unit_norm():
    e = HashEmbedder(embedding_dim=256)
    vecs = e.embed_batch(["Higgs boson LHC", "ATLAS CMS detector", "pasta carbonara"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def _embed_deterministic():
    e1, e2 = HashEmbedder(seed=7), HashEmbedder(seed=7)
    np.testing.assert_array_equal(e1.embed("test"), e2.embed("test"))

def _embed_similar_texts():
    e = HashEmbedder(embedding_dim=512)
    hep1 = e.embed("Higgs boson LHC ATLAS CMS discovery five sigma")
    hep2 = e.embed("Higgs boson mass 125 GeV electroweak symmetry breaking")
    cook = e.embed("pasta carbonara eggs cheese pepper Roman recipe")
    assert float(hep1 @ hep2) > float(hep1 @ cook), "related pair should score higher"

check("embed() shape",           _embed_shape)
check("unit-norm output",        _embed_unit_norm)
check("deterministic with seed", _embed_deterministic)
check("similar > dissimilar",    _embed_similar_texts)
check("factory hash backend",    lambda: get_embedder("hash", embedding_dim=32))

# ── NoiseFloor ───────────────────────────────────────────────────────────────
print(f"\n{HEAD}NoiseFloor{RST}")
from sigma_rag.noise_floor import NoiseFloor

def _make_embs(n=30, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    doc_ids = [f"doc_{i%3}" for i in range(n)]
    return vecs, doc_ids

def _nf_fit():
    nf = NoiseFloor()
    vecs, dids = _make_embs(30)
    nf.fit(vecs, doc_ids=dids, n_pairs=200, seed=0)
    assert np.isfinite(nf.mu_) and nf.sigma_ > 0

def _nf_threshold_formula():
    nf = NoiseFloor()
    vecs, dids = _make_embs(50)
    nf.fit(vecs, doc_ids=dids, n_pairs=300, seed=0)
    for n in [1.0, 2.0, 3.0]:
        expected = nf.mu_ + n * nf.sigma_
        assert abs(nf.threshold(n) - expected) < 1e-6

def _nf_not_fitted_raises():
    nf = NoiseFloor()
    try:
        nf.threshold(2.0)
        raise AssertionError("Should have raised")
    except RuntimeError:
        pass

def _nf_far_decreases():
    nf = NoiseFloor()
    vecs, dids = _make_embs(50)
    nf.fit(vecs, doc_ids=dids, n_pairs=300, seed=0)
    assert nf.false_alarm_rate(1.0) > nf.false_alarm_rate(2.0) > nf.false_alarm_rate(3.0)

check("fit() populates mu_, sigma_",      _nf_fit)
check("threshold() = mu_ + n*sigma_",     _nf_threshold_formula)
check("threshold() before fit raises",    _nf_not_fitted_raises)
check("FAR decreases with n_sigma",       _nf_far_decreases)

# ── SigmaIndex ───────────────────────────────────────────────────────────────
print(f"\n{HEAD}SigmaIndex{RST}")
from sigma_rag import SigmaIndex

DOCS = [
    ("The Higgs boson was discovered at the LHC in 2012 by ATLAS and CMS. "
     "Both experiments reported a local significance exceeding 5σ.", {"domain": "hep"}),
    ("A discovery in particle physics requires a local p-value below 2.87e-7, "
     "corresponding to a one-sided Gaussian significance of 5σ.", {"domain": "hep"}),
    ("Python is a high-level programming language. NumPy enables fast numerical computing.", {"domain": "code"}),
    ("The LHC accelerates protons to 6.8 TeV per beam. ATLAS and CMS are the two "
     "general-purpose detectors, each with tracking, calorimetry, and muon systems.", {"domain": "hep"}),
    ("The Standard Model classifies quarks, leptons, gauge bosons, and the Higgs boson. "
     "Three of four fundamental forces are unified under this framework.", {"domain": "hep"}),
    ("Dark matter comprises ~27% of the universe's energy content. Leading candidates "
     "include WIMPs and axions. Direct detection has not yet found a signal.", {"domain": "hep"}),
    ("Machine learning models learn patterns from training data. Deep neural networks "
     "use multiple layers of nonlinear transformations to extract hierarchical features.", {"domain": "ml"}),
    ("Quantum field theory describes particles as excitations of fields. The Standard "
     "Model is a gauge theory based on the symmetry group SU(3)×SU(2)×U(1).", {"domain": "hep"}),
    ("The look-elsewhere effect (LEE) accounts for scanning over multiple signal "
     "hypotheses. The global p-value is always larger than the local p-value.", {"domain": "hep"}),
    ("Quantum mechanics describes physics at the atomic scale. The Schrodinger equation "
     "governs the time evolution of a quantum state's probability amplitude.", {"domain": "hep"}),
    ("Pasta carbonara is a traditional Roman pasta dish. The sauce uses eggs, pecorino "
     "romano, guanciale, and black pepper. No cream is used in the original recipe.", {"domain": "cooking"}),
    ("Sourdough bread is leavened using wild yeast and lactic acid bacteria. The long cold "
     "fermentation develops complex flavours and a chewy crumb with an open structure.", {"domain": "cooking"}),
]

def _index_add_chunks():
    idx = SigmaIndex(embedder=HashEmbedder(), noise_n_pairs=50)
    idx.add_documents(DOCS)
    assert idx.n_chunks >= 3

def _index_calibrate():
    idx = SigmaIndex(embedder=HashEmbedder(), noise_n_pairs=100)
    idx.add_documents(DOCS)
    idx.calibrate(n_pairs=100, seed=0)
    assert idx.calibrated

def _index_check_ready_raises():
    idx = SigmaIndex(embedder=HashEmbedder(), noise_n_pairs=50)
    idx.add_documents(DOCS)
    try:
        idx.check_ready()
        raise AssertionError("Should have raised")
    except RuntimeError:
        pass

def _index_cosine_sim_shape():
    idx = SigmaIndex(embedder=HashEmbedder(), noise_n_pairs=100)
    idx.add_documents(DOCS)
    idx.calibrate(n_pairs=100, seed=0)
    q = idx.query_embeddings("Higgs boson LHC discovery")
    sims = idx.cosine_similarities(q)
    assert sims.shape == (idx.n_chunks,)

check("add_documents() produces chunks",    _index_add_chunks)
check("calibrate() sets calibrated=True",   _index_calibrate)
check("check_ready() raises uncalibrated",  _index_check_ready_raises)
check("cosine_similarities() shape",        _index_cosine_sim_shape)

# ── SigmaRetriever ───────────────────────────────────────────────────────────
print(f"\n{HEAD}SigmaRetriever{RST}")
from sigma_rag.retriever import SigmaRetriever, TopKRetriever

# Build a shared index
_idx = SigmaIndex(embedder=HashEmbedder(embedding_dim=512), noise_n_pairs=300)
_idx.add_documents(DOCS)
_idx.calibrate(n_pairs=300, seed=0)

def _sig_chunks_above_threshold():
    ret = SigmaRetriever(_idx, n_sigma=0.5, max_results=10)
    result = ret.retrieve("Higgs boson ATLAS CMS five sigma discovery")
    thr = _idx.noise_floor.threshold(result.n_sigma)
    for sc in result.significant:
        assert sc.similarity >= thr - 1e-6, f"sim={sc.similarity:.4f} < thr={thr:.4f}"

def _noise_chunks_below_threshold():
    ret = SigmaRetriever(_idx, n_sigma=0.5, max_results=10)
    result = ret.retrieve("Higgs boson LHC")
    thr = _idx.noise_floor.threshold(result.n_sigma)
    for sc in result.noise:
        assert sc.similarity < thr + 1e-6

def _max_results_respected():
    ret = SigmaRetriever(_idx, n_sigma=0.0, max_results=2)
    result = ret.retrieve("LHC proton collisions detector tracking calorimeter")
    assert len(result.significant) <= 2

def _topk_returns_k():
    ret = TopKRetriever(_idx, k=2)
    result = ret.retrieve("Higgs boson mass 125 GeV")
    assert len(result.significant) == 2

def _topk_always_has_evidence():
    ret = TopKRetriever(_idx, k=2)
    result = ret.retrieve("xyzzy nonsense frobnosticate quux")
    assert result.has_evidence is True

check("significant chunks ≥ threshold",    _sig_chunks_above_threshold)
check("noise chunks < threshold",           _noise_chunks_below_threshold)
check("max_results respected",              _max_results_respected)
check("TopK returns exactly k",            _topk_returns_k)
check("TopK always has_evidence=True",     _topk_always_has_evidence)

# ── SigmaRAGPipeline ─────────────────────────────────────────────────────────
print(f"\n{HEAD}SigmaRAGPipeline{RST}")
from sigma_rag import SigmaRAGPipeline
from sigma_rag.types import RAGResponse

_pipeline = SigmaRAGPipeline(_idx, n_sigma=0.5, llm="echo")

def _pipeline_returns_rag_response():
    r = _pipeline.query("Higgs boson LHC ATLAS CMS discovery significance")
    assert isinstance(r, RAGResponse)

def _pipeline_answer_is_string():
    r = _pipeline.query("local p-value background signal region sideband")
    assert isinstance(r.answer, str) and len(r.answer) > 0

def _pipeline_strict_no_evidence():
    strict = SigmaRAGPipeline(_idx, n_sigma=5.0, llm="echo")
    r = strict.query("pasta carbonara eggs cheese Roman recipe authentic")
    assert r.has_evidence is False
    assert r.context_used == ""

def _pipeline_compare_topk():
    comp = _pipeline.compare_with_topk("Higgs boson LHC discovery", k=2)
    assert "sigma_rag" in comp and "top_k" in comp

check("query() returns RAGResponse",          _pipeline_returns_rag_response)
check("response.answer is non-empty string",  _pipeline_answer_is_string)
check("strict threshold → no evidence",       _pipeline_strict_no_evidence)
check("compare_with_topk() both keys",        _pipeline_compare_topk)

# ── Summary ──────────────────────────────────────────────────────────────────
total   = len(results)
passed  = sum(1 for _, ok, _ in results if ok)
failed  = total - passed

print(f"\n{'─'*50}")
print(f"  {passed}/{total} checks passed", end="")
if failed:
    print(f"  ({FAIL} {failed} failed)")
    for name, ok, err in results:
        if not ok:
            print(f"    - {name}: {err}")
    sys.exit(1)
else:
    print(f"  {PASS}")
    print(f"  All σ-RAG modules validated offline ✓")
