"""
benchmark_beir.py
-----------------
Evaluate σ-RAG vs standard top-k retrieval on BEIR datasets.

Datasets are loaded directly from HuggingFace (no manual download).
Metrics: nDCG@10, Recall@100 — standard BEIR evaluation protocol.

Usage:
    pip install datasets sentence-transformers
    python benchmark_beir.py                         # scifact + nfcorpus
    python benchmark_beir.py --datasets scifact fiqa  # custom selection

Available datasets (small → large):
    nfcorpus (~3.6K docs), scifact (~5K), arguana (~8.7K),
    fiqa (~57K), quora (~523K)
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
# IR metrics
# ---------------------------------------------------------------------------


def dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def ndcg_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    """nDCG@k for a single query."""
    if not qrels:
        return 0.0
    rels = [qrels.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    ideal_dcg = dcg(ideal, k)
    return dcg(rels, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    """Recall@k — fraction of relevant docs found in top-k."""
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    found = sum(1 for d in retrieved[:k] if d in relevant)
    return found / len(relevant)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_beir_dataset(name: str) -> tuple[dict, dict, dict]:
    """
    Load a BEIR dataset from HuggingFace.

    Returns:
        corpus  : {doc_id: {"title": str, "text": str}}
        queries : {query_id: str}
        qrels   : {query_id: {doc_id: relevance_int}}
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the HuggingFace datasets library: pip install datasets") from exc

    print("  Loading corpus   ...", end=" ", flush=True)
    t0 = time.time()
    raw_corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus", trust_remote_code=True)
    corpus: dict[str, dict[str, str]] = {
        row["_id"]: {"title": row.get("title", ""), "text": row["text"]} for row in raw_corpus
    }
    print(f"{len(corpus):,} docs  ({time.time() - t0:.1f}s)")

    print("  Loading queries  ...", end=" ", flush=True)
    t0 = time.time()
    raw_queries = load_dataset(f"BeIR/{name}", "queries", split="queries", trust_remote_code=True)
    queries: dict[str, str] = {row["_id"]: row["text"] for row in raw_queries}
    print(f"{len(queries):,} queries  ({time.time() - t0:.1f}s)")

    print("  Loading qrels    ...", end=" ", flush=True)
    t0 = time.time()
    raw_qrels = load_dataset(f"BeIR/{name}-qrels", split="test", trust_remote_code=True)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in raw_qrels:
        qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    # Keep only queries that appear in qrels
    test_qids = set(qrels.keys())
    queries = {qid: q for qid, q in queries.items() if qid in test_qids}
    print(f"{len(queries):,} test queries  ({time.time() - t0:.1f}s)")

    return corpus, queries, dict(qrels)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def build_index(corpus: dict[str, dict[str, str]], chunk_size: int = 512) -> Any:
    """Index a BEIR corpus into SigmaIndex."""
    from sigma_rag import SigmaIndex

    index = SigmaIndex(chunk_size=chunk_size, chunk_overlap=64, n_sigma=2.0)

    doc_ids = list(corpus.keys())
    texts: list[str | tuple[str, dict]] = [
        f"{d['title']}. {d['text']}".strip() if d["title"] else d["text"] for d in corpus.values()
    ]

    print(f"  Embedding {len(texts):,} documents ...", end=" ", flush=True)
    t0 = time.time()
    index.add_documents(texts, doc_ids=doc_ids)
    print(f"{index.n_chunks:,} chunks  ({time.time() - t0:.1f}s)")

    print("  Calibrating noise floor ...", end=" ", flush=True)
    t0 = time.time()
    index.calibrate()
    print(f"done  ({time.time() - t0:.1f}s)")
    print(f"  {index.noise_floor.summary().splitlines()[1].strip()}")
    print(f"  {index.noise_floor.summary().splitlines()[2].strip()}")

    return index


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    index: Any,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    n_sigma: float = 2.0,
    k_topk: int = 10,
    k_eval: int = 10,
    recall_k: int = 100,
    max_queries: int = 500,
) -> dict[str, dict[str, float]]:
    """
    Run σ-RAG and top-k on all queries, return aggregated metrics.

    Args:
        index:      Calibrated SigmaIndex.
        queries:    {qid: text}
        qrels:      {qid: {doc_id: score}}
        n_sigma:    Significance threshold for σ-RAG.
        k_topk:     k for top-k baseline (retrieves exactly k).
        k_eval:     Rank cutoff for nDCG (nDCG@k_eval).
        recall_k:   Rank cutoff for Recall.
        max_queries: Cap for speed (BEIR test sets can be large).

    Returns:
        {method: {metric: value}}
    """
    from sigma_rag.retriever import SigmaRetriever, TopKRetriever

    sigma_ret = SigmaRetriever(index, n_sigma=n_sigma, max_results=recall_k)
    topk_ret = TopKRetriever(index, k=recall_k)

    qid_list = list(queries.keys())[:max_queries]
    n = len(qid_list)

    sigma_ndcg: list[float] = []
    sigma_recall: list[float] = []
    topk_ndcg: list[float] = []
    topk_recall: list[float] = []
    sigma_empty = 0  # queries where σ-RAG returned 0 results

    for i, qid in enumerate(qid_list):
        if i % 100 == 0:
            print(f"  [{i}/{n}] evaluating ...", end="\r", flush=True)

        q_text = queries[qid]
        q_qrels = qrels.get(qid, {})

        # σ-RAG: use doc_ids of significant chunks (deduplicated, order preserved)
        sr = sigma_ret.retrieve(q_text)
        sr_docs = _dedup_doc_ids([sc.chunk.doc_id for sc in sr.significant])
        if not sr_docs:
            sigma_empty += 1

        # Top-k
        tk = topk_ret.retrieve(q_text)
        tk_docs = _dedup_doc_ids([sc.chunk.doc_id for sc in tk.significant])

        sigma_ndcg.append(ndcg_at_k(sr_docs, q_qrels, k_eval))
        sigma_recall.append(recall_at_k(sr_docs, q_qrels, recall_k))
        topk_ndcg.append(ndcg_at_k(tk_docs, q_qrels, k_eval))
        topk_recall.append(recall_at_k(tk_docs, q_qrels, recall_k))

    print(f"  [{n}/{n}] done              ")

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "sigma_rag": {
            f"nDCG@{k_eval}": mean(sigma_ndcg),
            f"Recall@{recall_k}": mean(sigma_recall),
            "suppressed_%": sigma_empty / n * 100,
        },
        "top_k": {
            f"nDCG@{k_eval}": mean(topk_ndcg),
            f"Recall@{recall_k}": mean(topk_recall),
            "suppressed_%": 0.0,
        },
    }


def _dedup_doc_ids(doc_ids: list[str]) -> list[str]:
    """Remove duplicates while preserving rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_DEFAULTS = ["scifact", "nfcorpus"]


def run(datasets: list[str], n_sigma: float, k: int, max_queries: int) -> None:
    results: dict[str, dict[str, dict[str, float]]] = {}

    for name in datasets:
        print(f"\n{'=' * 65}")
        print(f"  Dataset: {name}")
        print(f"{'=' * 65}")

        corpus, queries, qrels = load_beir_dataset(name)
        index = build_index(corpus)

        print(f"\n  Running evaluation (n_sigma={n_sigma}, k={k}, max={max_queries} queries) ...")
        metrics = evaluate(
            index,
            queries,
            qrels,
            n_sigma=n_sigma,
            k_topk=k,
            k_eval=10,
            recall_k=100,
            max_queries=max_queries,
        )
        results[name] = metrics

        # Per-dataset table
        print(f"\n  {'Metric':<18}  {'σ-RAG':>10}  {'Top-k':>10}  {'Δ':>8}")
        print(f"  {'':─<18}  {'':─>10}  {'':─>10}  {'':─>8}")
        for metric in ["nDCG@10", "Recall@100"]:
            sv = metrics["sigma_rag"][metric]
            tv = metrics["top_k"][metric]
            delta = sv - tv
            sign = "+" if delta >= 0 else ""
            print(f"  {metric:<18}  {sv:>10.4f}  {tv:>10.4f}  {sign}{delta:>7.4f}")
        sup = metrics["sigma_rag"]["suppressed_%"]
        print(f"  {'Suppressed queries':<18}  {sup:>9.1f}%  {'—':>10}")

    # Cross-dataset summary
    if len(results) > 1:
        print(f"\n{'=' * 65}")
        print("  Summary across all datasets")
        print(f"{'=' * 65}")
        print(f"  {'Dataset':<14}  {'σ-RAG nDCG@10':>14}  {'Top-k nDCG@10':>14}  {'Δ':>8}")
        print(f"  {'':─<14}  {'':─>14}  {'':─>14}  {'':─>8}")
        for name, metrics in results.items():
            sv = metrics["sigma_rag"]["nDCG@10"]
            tv = metrics["top_k"]["nDCG@10"]
            delta = sv - tv
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<14}  {sv:>14.4f}  {tv:>14.4f}  {sign}{delta:>7.4f}")

    print(f"\n{'=' * 65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="σ-RAG BEIR benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_DEFAULTS,
        help="BEIR dataset names (default: scifact nfcorpus)",
    )
    parser.add_argument("--n_sigma", type=float, default=2.0, help="σ-RAG threshold")
    parser.add_argument("--k", type=int, default=100, help="Top-k baseline k")
    parser.add_argument(
        "--max_queries",
        type=int,
        default=500,
        help="Max test queries per dataset (for speed)",
    )
    args = parser.parse_args()
    run(args.datasets, args.n_sigma, args.k, args.max_queries)
