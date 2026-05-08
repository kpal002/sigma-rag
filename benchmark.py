"""
benchmark.py
------------
σ-RAG vs standard top-k retrieval benchmark.

Demonstrates the key advantage of significance-filtered retrieval:
when a question has NO relevant document in the corpus, top-k still
returns k results (forcing the LLM to hallucinate), while σ-RAG
correctly returns zero results and suppresses generation.

Usage:
    python benchmark.py

No API key required — uses sentence-transformers locally.
"""

import sys
import textwrap
from pathlib import Path

# Allow running from the outputs directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sigma_rag import SigmaIndex
from sigma_rag.retriever import SigmaRetriever, TopKRetriever


# ── Corpus ────────────────────────────────────────────────────────────────────
# Particle physics documents — relevant to HEP questions
PHYSICS_DOCS = [
    (
        "doc_higgs_discovery",
        """The Higgs boson was discovered at CERN's Large Hadron Collider in July 2012
        by the ATLAS and CMS experiments. Both experiments reported a local significance
        exceeding 5σ in the diphoton and four-lepton channels, satisfying the particle
        physics discovery threshold. The observed mass is approximately 125 GeV/c².
        The Higgs field breaks electroweak symmetry and gives mass to the W and Z bosons
        and to fermions through Yukawa couplings.""",
    ),
    (
        "doc_lhc_detector",
        """The Large Hadron Collider at CERN accelerates protons to 6.8 TeV per beam,
        producing 13.6 TeV centre-of-mass collisions in Run 3. The ATLAS detector is
        46 m long with a 25 m diameter; CMS uses a 3.8 T superconducting solenoid.
        Both detectors comprise silicon pixel and strip trackers, electromagnetic and
        hadronic calorimeters, and muon spectrometers. Trigger and data-acquisition
        systems select roughly 1 kHz of events from a 40 MHz collision rate.""",
    ),
    (
        "doc_significance",
        """In particle physics, a discovery requires a local p-value below 2.87 × 10⁻⁷,
        corresponding to a one-sided Gaussian significance of 5σ. The local p-value is
        the probability that a background-only fluctuation yields an excess at least as
        large as observed. Background is estimated from sidebands or control regions and
        extrapolated into the signal region. Evidence is declared at 3σ; discovery at 5σ.
        The look-elsewhere effect (LEE) corrects for scanning multiple mass hypotheses.""",
    ),
    (
        "doc_dm",
        """Dark matter comprises approximately 27% of the universe's total energy
        content. Evidence comes from galaxy rotation curves, gravitational lensing,
        and the cosmic microwave background power spectrum. Leading candidates include
        WIMPs (Weakly Interacting Massive Particles) and axions. Direct detection
        experiments such as LUX-ZEPLIN and XENONnT search for nuclear recoil
        signatures but have not yet observed a signal above background.""",
    ),
    (
        "doc_qft",
        """Quantum field theory combines quantum mechanics with special relativity.
        The Standard Model is a renormalisable gauge theory based on the symmetry group
        SU(3) × SU(2) × U(1), describing the strong, weak, and electromagnetic forces.
        Particles are excitations of underlying quantum fields. Perturbative calculations
        using Feynman diagrams agree with experimental measurements to many decimal
        places, making the Standard Model the most precisely tested theory in physics.""",
    ),
]

# Completely unrelated documents — cooking domain
COOKING_DOCS = [
    (
        "doc_bread",
        """Sourdough bread is made through natural fermentation using wild yeast and
        lactic acid bacteria. The starter culture must be regularly fed with flour and
        water. Bulk fermentation typically takes 4-12 hours at room temperature. The
        characteristic sour flavour comes from acetic and lactic acids produced during
        fermentation. High hydration doughs (75-80%) produce an open crumb structure.""",
    ),
    (
        "doc_pasta",
        """Fresh pasta is made from semolina flour and eggs. The dough must be kneaded
        for 10 minutes until smooth and elastic, then rested for 30 minutes. Rolling
        to 1-2mm thickness produces the best texture. Fresh pasta cooks in just 2-3
        minutes in boiling salted water, much faster than dried pasta.""",
    ),
]


# ── Benchmark questions ───────────────────────────────────────────────────────
QUESTIONS = [
    # Answerable questions (relevant docs exist)
    {
        "q": "What significance level was required to claim the Higgs discovery?",
        "answerable": True,
        "relevant_docs": ["doc_higgs_discovery", "doc_significance"],
    },
    {
        "q": "What is the mass of the Higgs boson?",
        "answerable": True,
        "relevant_docs": ["doc_higgs_discovery"],
    },
    {
        "q": "What evidence supports the existence of dark matter?",
        "answerable": True,
        "relevant_docs": ["doc_dm"],
    },
    # Unanswerable questions (NO relevant doc in corpus — critical test)
    {
        "q": "What is the best recipe for chocolate cake?",
        "answerable": False,
        "relevant_docs": [],
    },
    {
        "q": "How do you treat type 2 diabetes with medication?",
        "answerable": False,
        "relevant_docs": [],
    },
    {
        "q": "What are the rules of cricket?",
        "answerable": False,
        "relevant_docs": [],
    },
]


# ── Metrics helpers ────────────────────────────────────────────────────────────

def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Fraction of retrieved docs that are relevant."""
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d in relevant_doc_ids)
    return hits / len(retrieved_doc_ids)


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Fraction of relevant docs that were retrieved."""
    if not relevant_doc_ids:
        return 1.0  # vacuously true if nothing is relevant
    hits = sum(1 for d in retrieved_doc_ids if d in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def hallucination_rate(results: list[dict]) -> float:
    """
    Fraction of unanswerable questions where the retriever
    returned at least one chunk (hallucination risk).
    """
    unanswerable = [r for r in results if not r["answerable"]]
    if not unanswerable:
        return 0.0
    risky = sum(1 for r in unanswerable if r["n_retrieved"] > 0)
    return risky / len(unanswerable)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(n_sigma: float = 2.0, k: int = 3) -> None:
    """
    Run the full benchmark and print a comparison table.

    Args:
        n_sigma: Significance threshold for σ-RAG.
        k:       Number of results for top-k baseline.
    """
    print("=" * 70)
    print("  σ-RAG Benchmark: Significance-Filtered vs Standard Top-k Retrieval")
    print("=" * 70)

    # ── Build index ───────────────────────────────────────────────────
    print("\n[1/3] Building index ...")
    index = SigmaIndex(n_sigma=n_sigma, chunk_size=400, chunk_overlap=50)

    all_docs = PHYSICS_DOCS + COOKING_DOCS
    texts = [text for _, text in all_docs]
    doc_ids = [did for did, _ in all_docs]
    index.add_documents(texts, doc_ids=doc_ids)
    index.calibrate(n_pairs=2000)  # small corpus → fewer pairs needed

    print(index.noise_floor.summary())

    # ── Instantiate retrievers ────────────────────────────────────────
    sigma_ret = SigmaRetriever(index, n_sigma=n_sigma, max_results=k)
    topk_ret = TopKRetriever(index, k=k)

    # ── Run questions ─────────────────────────────────────────────────
    print(f"\n[2/3] Running {len(QUESTIONS)} benchmark questions ...\n")

    sigma_results = []
    topk_results = []

    col_w = 42
    print(f"{'Question':<{col_w}}  {'σ-RAG':^20}  {'Top-k':^20}")
    print(f"{'':─<{col_w}}  {'':─<20}  {'':─<20}")

    for item in QUESTIONS:
        q = item["q"]
        relevant = item["relevant_docs"]
        answerable = item["answerable"]

        # σ-RAG
        sr = sigma_ret.retrieve(q)
        sr_doc_ids = [sc.chunk.doc_id for sc in sr.significant]
        sr_prec = precision_at_k(sr_doc_ids, relevant)
        sr_n = len(sr.significant)
        sigma_results.append({
            "q": q, "answerable": answerable,
            "n_retrieved": sr_n, "precision": sr_prec,
            "relevant": relevant,
        })

        # Top-k
        tk = topk_ret.retrieve(q)
        tk_doc_ids = [sc.chunk.doc_id for sc in tk.significant]
        tk_prec = precision_at_k(tk_doc_ids, relevant)
        tk_n = len(tk.significant)
        topk_results.append({
            "q": q, "answerable": answerable,
            "n_retrieved": tk_n, "precision": tk_prec,
            "relevant": relevant,
        })

        # Format row
        q_short = textwrap.shorten(q, width=col_w, placeholder="...")
        answerable_tag = "✓" if answerable else "✗ (unanswerable)"

        sr_col = (
            f"{sr_n} chunks  P={sr_prec:.2f}"
            if sr_n > 0
            else "⛔ 0 chunks (correct!)"
        )
        tk_col = f"{tk_n} chunks  P={tk_prec:.2f}"

        print(f"{q_short:<{col_w}}  {sr_col:^20}  {tk_col:^20}  [{answerable_tag}]")

    # ── Summary statistics ────────────────────────────────────────────
    print(f"\n[3/3] Summary\n{'─'*60}")

    answerable_q = [r for r in QUESTIONS if r["answerable"]]
    unanswerable_q = [r for r in QUESTIONS if not r["answerable"]]

    # Precision on answerable questions
    sr_precs = [r["precision"] for r in sigma_results if r["answerable"]]
    tk_precs = [r["precision"] for r in topk_results if r["answerable"]]

    print(f"\nAnswerable questions ({len(answerable_q)} total):")
    print(f"  σ-RAG  avg precision : {np.mean(sr_precs):.3f}")
    print(f"  Top-k  avg precision : {np.mean(tk_precs):.3f}")

    print(f"\nUnanswerable questions ({len(unanswerable_q)} total):")
    sr_hal = hallucination_rate(sigma_results)
    tk_hal = hallucination_rate(topk_results)
    print(f"  σ-RAG  hallucination risk : {sr_hal:.0%}  (0% = no noisy context injected ✓)")
    print(f"  Top-k  hallucination risk : {tk_hal:.0%}  (100% = always injects irrelevant context ✗)")

    print(f"\nNoise floor: μ={index.noise_floor.mu_:.4f}, σ={index.noise_floor.sigma_:.4f}")
    print(f"Threshold @ {n_sigma}σ: {index.noise_floor.threshold(n_sigma):.4f}")
    print(f"False alarm rate @ {n_sigma}σ: {index.noise_floor.false_alarm_rate(n_sigma):.2%}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmark(n_sigma=2.0, k=3)
