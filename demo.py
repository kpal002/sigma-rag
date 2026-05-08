"""
demo.py
-------
Interactive demo of σ-RAG.

Shows three scenarios that highlight σ-RAG's advantage over standard RAG:
  1. A question with a clear, relevant answer in the corpus.
  2. A question partially addressed by the corpus.
  3. A question with NO relevant document — σ-RAG suppresses generation.

Usage:
    python demo.py                      # offline, no LLM generation
    python demo.py --llm anthropic      # with Anthropic generation
    python demo.py --llm openai         # with OpenAI generation

No API key required for the default (offline) run.
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sigma_rag import SigmaIndex, SigmaRAGPipeline
from sigma_rag.retriever import SigmaRetriever, TopKRetriever


# ── Sample corpus ─────────────────────────────────────────────────────────────

CORPUS = {
    "higgs_discovery": """
        The Higgs boson was discovered at CERN's Large Hadron Collider in July 2012 by
        the ATLAS and CMS experiments. Both experiments observed an excess of events
        over the expected background in the diphoton and four-lepton decay channels,
        each reaching a local significance of 5σ — the standard threshold for a
        discovery claim in particle physics. The observed mass was approximately
        125 GeV/c². The Higgs field is responsible for electroweak symmetry breaking
        and gives mass to the W and Z bosons and to fermions via Yukawa couplings.
    """,
    "lhc_detector": """
        The LHC at CERN accelerates protons to 6.8 TeV per beam, producing
        centre-of-mass collision energies of 13.6 TeV in Run 3. The ATLAS detector
        is 46 m long and 25 m in diameter; CMS uses a 3.8 T superconducting solenoid.
        Key sub-systems include silicon tracking detectors, electromagnetic and hadronic
        calorimeters, and muon spectrometers. The principal backgrounds at the LHC are
        QCD multijet production, W/Z+jets, and top-quark pair production.
    """,
    "signal_significance": """
        In particle physics, a discovery requires a local p-value below 2.87 × 10⁻⁷,
        corresponding to a one-sided Gaussian significance of 5σ. The local p-value
        measures the probability that a background-only fluctuation could produce an
        excess at least as large as observed. The background is estimated from
        sidebands or control regions, then extrapolated into the signal region.
        "Evidence" is declared at 3σ; "observation" or "discovery" at 5σ.
    """,
    "standard_model": """
        The Standard Model of particle physics describes three fundamental forces
        (electromagnetic, weak, and strong nuclear) and classifies all known elementary
        particles. It contains 17 fundamental particles: 6 quarks, 6 leptons, 4 gauge
        bosons, and the Higgs boson. The model has been tested to extraordinary
        precision, with some predictions verified to 10 decimal places. It does not
        include gravity or explain dark matter and dark energy.
    """,
    "python_basics": """
        Python is a high-level, interpreted programming language emphasising readability.
        It supports multiple programming paradigms: procedural, object-oriented, and
        functional. Key features include dynamic typing, automatic memory management,
        and a large standard library. Python 3.12 introduced significant performance
        improvements including a faster interpreter and reduced startup times.
    """,
}


# ── Demo scenarios ─────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "title": "Scenario 1 — Strong match (clear answer in corpus)",
        "question": "What significance level was required to claim the Higgs discovery?",
        "expected": "Strong retrieval — clear answer exists in signal_significance doc.",
    },
    {
        "title": "Scenario 2 — Moderate match (partial answer)",
        "question": "What are the main background processes at the LHC?",
        "expected": "Should retrieve lhc_detector doc.",
    },
    {
        "title": "Scenario 3 — No match (unanswerable — KEY DEMO)",
        "question": "What is the best treatment for hypertension?",
        "expected": "σ-RAG: 0 chunks. Top-k: still returns 3 irrelevant chunks.",
    },
    {
        "title": "Scenario 4 — Adjacent domain (Python in a particle physics corpus)",
        "question": "How do I install Python packages with pip?",
        "expected": "Python doc is in corpus but unrelated to the physics domain.",
    },
]


# ── Display helpers ────────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
DIM = "\033[2m"


def print_header(text: str) -> None:
    """Print a bold section header."""
    print(f"\n{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")


def print_retrieval_row(label: str, chunks, threshold: float, is_sigma: bool) -> None:
    """Print one row of retrieval results."""
    n = len(chunks)
    colour = GREEN if (is_sigma and n == 0) else (RED if (is_sigma and n == 0) else RESET)

    print(f"\n  {BOLD}{label}{RESET}")
    if not chunks:
        if is_sigma:
            print(f"    {GREEN}⛔ 0 chunks returned — suppressing generation (correct!){RESET}")
        else:
            print(f"    {RED}⚠  0 chunks returned{RESET}")
        return

    for i, sc in enumerate(chunks, 1):
        flag = (
            f"{GREEN}✓ significant{RESET}"
            if sc.significant
            else f"{YELLOW}~ below threshold{RESET}"
        )
        sim_bar = "█" * int(sc.similarity * 20)
        print(
            f"    [{i}] z={sc.z_score:+.2f}σ  sim={sc.similarity:.4f}  "
            f"p={sc.p_value:.4f}  {flag}"
        )
        preview = textwrap.shorten(sc.chunk.text.strip(), width=70, placeholder="...")
        print(f"        {DIM}{preview}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(llm_backend: str = "echo") -> None:
    """
    Run the full demo.

    Args:
        llm_backend: 'echo', 'anthropic', or 'openai'.
    """
    print_header("σ-RAG Interactive Demo")
    print(
        f"  Corpus: {len(CORPUS)} documents | "
        f"LLM backend: {llm_backend}"
    )

    # ── Build index ────────────────────────────────────────────────────
    print("\n  Building index ...")
    index = SigmaIndex(n_sigma=2.0, chunk_size=300, chunk_overlap=50)
    index.add_documents(
        list(CORPUS.values()),
        doc_ids=list(CORPUS.keys()),
    )
    index.calibrate(n_pairs=1000)

    print(f"\n  {BOLD}Noise Floor{RESET}")
    for line in index.noise_floor.summary().splitlines():
        print(f"    {line}")

    # ── Retrievers ─────────────────────────────────────────────────────
    sigma_ret = SigmaRetriever(index, n_sigma=2.0, max_results=3)
    topk_ret = TopKRetriever(index, k=3)

    # Optionally set up pipeline for LLM generation
    pipeline = None
    if llm_backend != "echo":
        pipeline = SigmaRAGPipeline(index, llm=llm_backend, n_sigma=2.0)

    # ── Run scenarios ──────────────────────────────────────────────────
    for scenario in SCENARIOS:
        print_header(scenario["title"])
        q = scenario["question"]
        print(f"  {BOLD}Question:{RESET} {q}")
        print(f"  {DIM}Expected: {scenario['expected']}{RESET}")

        # σ-RAG retrieval
        sr = sigma_ret.retrieve(q)
        print_retrieval_row(
            f"σ-RAG (threshold={sr.threshold:.4f} @ {sr.n_sigma}σ)",
            sr.significant,
            sr.threshold,
            is_sigma=True,
        )

        # Top-k retrieval
        tk = topk_ret.retrieve(q)
        print_retrieval_row(
            f"Top-k (k=3, no threshold)",
            tk.significant,
            tk.threshold,
            is_sigma=False,
        )

        # LLM generation (if available)
        if pipeline is not None:
            print(f"\n  {BOLD}σ-RAG Answer:{RESET}")
            resp = pipeline.query(q)
            for line in textwrap.wrap(resp.answer, width=64):
                print(f"    {line}")

    # ── Final summary ──────────────────────────────────────────────────
    print_header("Key Insight")
    print(
        "  Standard top-k ALWAYS returns k chunks, even for completely\n"
        "  irrelevant queries. This floods the LLM with noisy context,\n"
        "  leading to hallucinated answers.\n\n"
        "  σ-RAG sets a significance threshold based on the BACKGROUND\n"
        "  distribution of the embedding space (analogous to the local\n"
        "  p-value threshold in a particle physics signal search — 5σ\n"
        "  for discovery at the LHC). Chunks below the threshold are\n"
        "  discarded — preventing background contamination of context.\n\n"
        f"  False-alarm rate @ 2σ: "
        f"{index.noise_floor.false_alarm_rate(2.0):.2%}  "
        f"(~1 in 43 random pairs is a false positive)\n"
        f"  False-alarm rate @ 3σ: "
        f"{index.noise_floor.false_alarm_rate(3.0):.3%}  "
        f"(~1 in 741 random pairs)"
    )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="σ-RAG Demo")
    parser.add_argument(
        "--llm",
        choices=["echo", "anthropic", "openai"],
        default="echo",
        help="LLM backend for generation (default: echo, no API key needed)",
    )
    args = parser.parse_args()
    run_demo(llm_backend=args.llm)
