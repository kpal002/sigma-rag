"""
benchmark_beir.py
-----------------
Demonstrate σ-RAG's hallucination-prevention value on BEIR datasets.

Design
------
  1. Index the corpus of one BEIR dataset (the "in-domain" corpus).
  2. Answerable queries  — in-domain test queries that have ground-truth
     relevant documents in the corpus.  Both σ-RAG and top-k are scored
     on nDCG@10 (retrieval quality should be comparable).
  3. Unanswerable queries — test queries from a *different* domain dataset.
     By construction, none of their relevant documents exist in the indexed
     corpus, so any retrieved chunk is noise and any generated answer is a
     hallucination.

Key metrics
-----------
  Answerable    : nDCG@10  (σ-RAG should match top-k)
  Unanswerable  : suppression rate  (σ-RAG should abstain; top-k never does)

Usage
-----
    pip install datasets sentence-transformers
    python benchmark_beir.py
    python benchmark_beir.py --corpus scifact --ood fiqa --n_sigma 2.0

The default pairing (scifact corpus + nfcorpus OOD queries) runs in
~3 minutes on a laptop CPU with sentence-transformers.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import defaultdict
from typing import Any

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# IR metrics (answerable side)
# ---------------------------------------------------------------------------


def dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def ndcg_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    if not qrels:
        return 0.0
    rels = [qrels.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    ideal_dcg = dcg(ideal, k)
    return dcg(rels, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_corpus_and_queries(name: str) -> tuple[dict, dict, dict]:
    """
    Load corpus, queries, and test qrels for a BEIR dataset.

    Returns:
        corpus  : {doc_id: {"title": str, "text": str}}
        queries : {query_id: str}   — test queries only
        qrels   : {query_id: {doc_id: int}}
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    print("  corpus   ...", end=" ", flush=True)
    t0 = time.time()
    raw = load_dataset(f"BeIR/{name}", "corpus", split="corpus", trust_remote_code=True)
    corpus = {str(r["_id"]): {"title": r.get("title", ""), "text": r["text"]} for r in raw}
    print(f"{len(corpus):,} docs  ({time.time() - t0:.1f}s)")

    print("  queries  ...", end=" ", flush=True)
    t0 = time.time()
    raw = load_dataset(f"BeIR/{name}", "queries", split="queries", trust_remote_code=True)
    all_queries = {str(r["_id"]): r["text"] for r in raw}
    print(f"{len(all_queries):,} total  ({time.time() - t0:.1f}s)")

    print("  qrels    ...", end=" ", flush=True)
    t0 = time.time()
    raw = load_dataset(f"BeIR/{name}-qrels", split="test", trust_remote_code=True)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in raw:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    test_qids = set(qrels.keys())
    queries = {qid: q for qid, q in all_queries.items() if qid in test_qids}
    print(f"{len(queries):,} test queries  ({time.time() - t0:.1f}s)")

    return corpus, queries, dict(qrels)


def load_ood_queries(name: str, max_queries: int) -> list[str]:
    """
    Load test queries from a different (out-of-domain) BEIR dataset.
    These have no relevant documents in the in-domain corpus by construction.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    print("  OOD queries ...", end=" ", flush=True)
    t0 = time.time()
    # Get query IDs that appear in the test qrels (so they are real test queries)
    raw_qrels = load_dataset(f"BeIR/{name}-qrels", split="test", trust_remote_code=True)
    test_qids = {str(r["query-id"]) for r in raw_qrels}

    raw_q = load_dataset(f"BeIR/{name}", "queries", split="queries", trust_remote_code=True)
    queries = [r["text"] for r in raw_q if str(r["_id"]) in test_qids][:max_queries]
    print(f"{len(queries):,} queries from '{name}'  ({time.time() - t0:.1f}s)")
    return queries


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_index(corpus: dict, chunk_size: int = 512) -> Any:
    from sigma_rag import SigmaIndex

    index = SigmaIndex(chunk_size=chunk_size, chunk_overlap=64, n_sigma=2.0)
    doc_ids = list(corpus.keys())
    texts: list[str | tuple[str, dict]] = [
        f"{d['title']}. {d['text']}".strip() if d["title"] else d["text"] for d in corpus.values()
    ]

    print(f"  embedding {len(texts):,} docs ...", end=" ", flush=True)
    t0 = time.time()
    index.add_documents(texts, doc_ids=doc_ids)
    print(f"{index.n_chunks:,} chunks  ({time.time() - t0:.1f}s)")

    print("  calibrating noise floor ...", end=" ", flush=True)
    t0 = time.time()
    index.calibrate()
    print(f"done  ({time.time() - t0:.1f}s)")

    nf = index.noise_floor
    print(f"  μ={nf.mu_:.4f}  σ={nf.sigma_:.4f}  threshold@2σ={nf.threshold(2.0):.4f}")
    return index


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _dedup(doc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def eval_answerable(
    index: Any,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    n_sigma: float,
    k: int,
    max_q: int,
) -> dict[str, dict[str, float]]:
    """Score retrieval quality on in-domain (answerable) queries."""
    from sigma_rag.retriever import SigmaRetriever, TopKRetriever

    sigma_ret = SigmaRetriever(index, n_sigma=n_sigma, max_results=k)
    topk_ret = TopKRetriever(index, k=k)

    qid_list = list(queries.keys())[:max_q]
    n = len(qid_list)
    if n == 0:
        raise ValueError("No answerable test queries found.")

    s_ndcg, s_rec, t_ndcg, t_rec = [], [], [], []
    s_suppressed = 0

    for i, qid in enumerate(qid_list):
        if i % 100 == 0:
            print(f"  [{i}/{n}] ...", end="\r", flush=True)
        q_qrels = qrels.get(qid, {})
        qt = queries[qid]

        sr = sigma_ret.retrieve(qt)
        sr_docs = _dedup([sc.chunk.doc_id for sc in sr.significant])
        if not sr_docs:
            s_suppressed += 1

        tk = topk_ret.retrieve(qt)
        tk_docs = _dedup([sc.chunk.doc_id for sc in tk.significant])

        s_ndcg.append(ndcg_at_k(sr_docs, q_qrels, 10))
        s_rec.append(recall_at_k(sr_docs, q_qrels, k))
        t_ndcg.append(ndcg_at_k(tk_docs, q_qrels, 10))
        t_rec.append(recall_at_k(tk_docs, q_qrels, k))

    print(f"  [{n}/{n}] done              ")

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    return {
        "sigma_rag": {
            "nDCG@10": mean(s_ndcg),
            f"Recall@{k}": mean(s_rec),
            "suppressed_%": s_suppressed / n * 100,
            "n_queries": n,
        },
        "top_k": {
            "nDCG@10": mean(t_ndcg),
            f"Recall@{k}": mean(t_rec),
            "suppressed_%": 0.0,
            "n_queries": n,
        },
    }


def eval_unanswerable(
    index: Any,
    ood_queries: list[str],
    n_sigma: float,
    k: int,
) -> dict[str, dict[str, float]]:
    """
    Measure hallucination risk on out-of-domain (unanswerable) queries.

    top-k always retrieves k chunks → 100% hallucination risk.
    σ-RAG suppresses queries below the significance threshold → lower risk.
    """
    from sigma_rag.retriever import SigmaRetriever

    sigma_ret = SigmaRetriever(index, n_sigma=n_sigma, max_results=k)
    n = len(ood_queries)
    s_suppressed = 0

    for i, qt in enumerate(ood_queries):
        if i % 100 == 0:
            print(f"  [{i}/{n}] ...", end="\r", flush=True)
        sr = sigma_ret.retrieve(qt)
        if not sr.significant:
            s_suppressed += 1

    print(f"  [{n}/{n}] done              ")

    suppression_rate = s_suppressed / n * 100
    hallucination_risk_sigma = 100 - suppression_rate

    return {
        "sigma_rag": {
            "suppression_%": suppression_rate,
            "hallucination_risk_%": hallucination_risk_sigma,
            "n_queries": n,
        },
        "top_k": {
            "suppression_%": 0.0,
            "hallucination_risk_%": 100.0,
            "n_queries": n,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    corpus_dataset: str,
    ood_dataset: str,
    n_sigma: float,
    k: int,
    max_answerable: int,
    max_unanswerable: int,
) -> None:
    W = 65
    print(f"\n{'=' * W}")
    print("  σ-RAG Hallucination-Prevention Benchmark")
    print(f"  Corpus: {corpus_dataset}  |  OOD queries: {ood_dataset}  |  n_sigma={n_sigma}")
    print(f"{'=' * W}")

    # ── Load corpus + in-domain queries ──────────────────────────────
    print(f"\n[1/4] Loading in-domain dataset: {corpus_dataset}")
    corpus, id_queries, id_qrels = load_corpus_and_queries(corpus_dataset)

    # ── Load OOD queries ─────────────────────────────────────────────
    print(f"\n[2/4] Loading out-of-domain queries: {ood_dataset}")
    ood_queries = load_ood_queries(ood_dataset, max_unanswerable)

    # ── Build index ───────────────────────────────────────────────────
    print(f"\n[3/4] Building index from {corpus_dataset} corpus")
    index = build_index(corpus)

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n[4/4] Evaluating ...")

    print(f"\n  --- Answerable queries (in-domain, n≤{max_answerable}) ---")
    ans_metrics = eval_answerable(index, id_queries, id_qrels, n_sigma, k, max_answerable)

    print(f"\n  --- Unanswerable queries (out-of-domain, n={len(ood_queries)}) ---")
    unans_metrics = eval_unanswerable(index, ood_queries, n_sigma, k)

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  RESULTS")
    print(f"{'=' * W}")

    am_s = ans_metrics["sigma_rag"]
    am_t = ans_metrics["top_k"]
    n_ans = int(am_s["n_queries"])
    n_unans = int(unans_metrics["sigma_rag"]["n_queries"])

    print(f"\n  Answerable queries ({n_ans} queries — retrieval quality)")
    print(f"  {'Metric':<20}  {'σ-RAG':>10}  {'Top-k':>10}  {'Δ':>8}")
    print(f"  {'':─<20}  {'':─>10}  {'':─>10}  {'':─>8}")
    for metric in ["nDCG@10", f"Recall@{k}"]:
        sv, tv = am_s[metric], am_t[metric]
        delta = sv - tv
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<20}  {sv:>10.4f}  {tv:>10.4f}  {sign}{delta:>7.4f}")
    sup = am_s["suppressed_%"]
    print(f"  {'Suppressed':<20}  {sup:>9.1f}%  {'0.0%':>10}")

    um_s = unans_metrics["sigma_rag"]
    um_t = unans_metrics["top_k"]
    print(f"\n  Unanswerable queries ({n_unans} queries — hallucination prevention)")
    print(f"  {'Metric':<26}  {'σ-RAG':>10}  {'Top-k':>10}")
    print(f"  {'':─<26}  {'':─>10}  {'':─>10}")
    print(
        f"  {'Correctly suppressed':<26}  {um_s['suppression_%']:>9.1f}%  {'0.0%':>10}  ← σ-RAG abstains"
    )
    print(
        f"  {'Hallucination risk':<26}  {um_s['hallucination_risk_%']:>9.1f}%  {um_t['hallucination_risk_%']:>9.1f}%  ← top-k always retrieves"
    )

    reduction = um_t["hallucination_risk_%"] - um_s["hallucination_risk_%"]
    print(f"\n  ✓ σ-RAG reduces hallucination risk by {reduction:.1f} percentage points")
    print(
        f"    while matching top-k nDCG@10 within {abs(am_s['nDCG@10'] - am_t['nDCG@10']):.4f} on answerable queries."
    )
    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="σ-RAG hallucination-prevention benchmark")
    parser.add_argument(
        "--corpus", default="scifact", help="In-domain BEIR dataset (default: scifact)"
    )
    parser.add_argument(
        "--ood", default="nfcorpus", help="Out-of-domain query dataset (default: nfcorpus)"
    )
    parser.add_argument("--n_sigma", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=10, help="Retrieval depth (default: 10)")
    parser.add_argument("--max_answerable", type=int, default=300)
    parser.add_argument("--max_unanswerable", type=int, default=300)
    args = parser.parse_args()
    run(args.corpus, args.ood, args.n_sigma, args.k, args.max_answerable, args.max_unanswerable)
