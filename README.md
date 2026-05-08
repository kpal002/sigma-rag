# σ-RAG · Sigma-RAG

[![PyPI version](https://img.shields.io/pypi/v/sigma-rag.svg)](https://pypi.org/project/sigma-rag/)
[![Python](https://img.shields.io/pypi/pyversions/sigma-rag.svg)](https://pypi.org/project/sigma-rag/)
[![CI](https://github.com/kpal002/sigma-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kpal002/sigma-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Stop injecting noise into your LLM context.**
> σ-RAG gates retrieval with a statistical significance threshold so your model only sees chunks that are *actually* relevant — not just the least-bad ones.

---

## The Problem with Standard RAG

Standard RAG always returns the **top-k chunks**, regardless of whether any of them are relevant to the query.

```
Query: "What caused the 2008 financial crisis?"
Corpus: Python tutorials, particle-physics papers, cooking recipes

Top-3 RAG returns:  chunk_47 (sim=0.31), chunk_12 (sim=0.29), chunk_89 (sim=0.28)
                    ← ALL noise. LLM hallucinates an answer anyway.

σ-RAG returns:      ⚠️  No significant evidence found. Response suppressed.
                    ← Hallucination prevented.
```

When no chunk is relevant, top-k RAG silently feeds the LLM garbage context. The LLM, trained to be helpful, fabricates a plausible-sounding answer. **σ-RAG breaks this failure mode.**

---

## How It Works

σ-RAG characterises the **noise floor** of your embedding space — the distribution of cosine similarities between *random, unrelated document pairs*. This is analogous to estimating the background noise level before declaring a signal detection.

```
1. Sample N random cross-document pairs from your corpus
2. Fit a Gaussian: μ_noise, σ_noise
3. Threshold = μ_noise + n·σ_noise   (default n=2, FAR ≈ 2.3%)
4. At query time: only chunks with similarity > threshold are "significant"
5. If zero chunks clear the bar → suppress generation entirely
```

The threshold has a principled interpretation: at `n=2σ`, the false alarm rate (probability a random noise chunk clears the bar) is ≈ 2.3%. At `n=3σ`, it drops to ≈ 0.13%.

---

## Benchmark

Evaluated on a mixed corpus (physics papers + cooking articles) with answerable and unanswerable questions:

| Metric | Standard Top-3 | σ-RAG (2σ) |
|--------|:--------------:|:----------:|
| Precision@3 (answerable) | 1.00 | 1.00 |
| Recall@3 (answerable) | 1.00 | 0.95 |
| **Hallucination risk** (unanswerable) | **100%** | **0%** |
| Avg chunks passed to LLM | 3.0 | 1.8 |

σ-RAG matches top-k on answerable questions while **eliminating hallucination risk on unanswerable ones**.

---

## Installation

```bash
# Minimal (numpy only — uses HashEmbedder, good for testing)
pip install sigma-rag

# Recommended (local sentence-transformers embeddings)
pip install "sigma-rag[local]"

# With Anthropic LLM backend
pip install "sigma-rag[local,anthropic]"

# Everything
pip install "sigma-rag[all]"
```

---

## Quick Start

```python
from sigma_rag import SigmaIndex, SigmaRAGPipeline

# 1. Build the index
index = SigmaIndex()
index.add_documents([
    "The Higgs boson was discovered at the LHC in 2012 by ATLAS and CMS at 5σ significance...",
    "A discovery in particle physics requires a local p-value below 2.87e-7 (5σ)...",
    "The Standard Model describes quarks, leptons, gauge bosons, and the Higgs field...",
])
index.calibrate()   # fits the background distribution

# 2. Query (offline echo mode — no API key needed)
pipeline = SigmaRAGPipeline(index, llm="echo")

# Answerable query → returns answer
response = pipeline.query("What significance was required to claim the Higgs discovery?")
print(response.has_evidence)     # True
print(f"Used {len(response.retrieval.significant)} chunks")

# Unanswerable query → suppressed
response = pipeline.query("What is the best pasta carbonara recipe?")
print(response.has_evidence)     # False  ← hallucination prevented
print(response.answer)           # "⚠️  σ-RAG: No significant evidence..."
```

---

## API Overview

### `SigmaIndex`

```python
index = SigmaIndex(
    chunk_size=512,       # max chars per chunk
    chunk_overlap=64,     # overlap between consecutive chunks
    n_sigma=2.0,          # default significance threshold
)
index.add_documents(docs)   # list of strings or (text, metadata) tuples
index.calibrate()            # REQUIRED before querying
```

### `SigmaRAGPipeline`

```python
pipeline = SigmaRAGPipeline(
    index,
    n_sigma=2.0,           # threshold (override per-query with pipeline.query(..., n_sigma=3.0))
    max_results=5,         # max chunks to pass to LLM
    llm="anthropic",       # "anthropic" | "openai" | "echo"
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
)
response = pipeline.query("Your question here")
```

### `RAGResponse` fields

```python
response.answer           # str — the answer (or suppression message)
response.has_evidence     # bool — False means generation was suppressed
response.retrieval        # RetrievalResult with .significant and .noise lists
response.retrieval.significant[0].z_score    # how many σ above noise floor
response.retrieval.significant[0].p_value    # probability under null
```

### Side-by-side comparison

```python
comparison = pipeline.compare_with_topk("What is dark matter?", k=5)
print(comparison["sigma_rag"].answer)
print(comparison["top_k"].answer)
```

---

## Embedder Backends

| Embedder | Install | Quality | API Key |
|----------|---------|---------|---------|
| `HashEmbedder` | built-in | Testing only | No |
| `SentenceTransformerEmbedder` | `pip install "sigma-rag[local]"` | Good | No |
| `OpenAIEmbedder` | `pip install "sigma-rag[openai]"` | Excellent | Yes |

```python
from sigma_rag import SigmaIndex, OpenAIEmbedder

index = SigmaIndex(embedder=OpenAIEmbedder(model="text-embedding-3-large"))
```

---

## Adjusting the Threshold

```python
# More permissive: catch more relevant chunks, higher false-alarm rate
response = pipeline.query(question, n_sigma=1.5)   # FAR ≈ 6.7%

# More conservative: fewer false positives, may miss weak signals
response = pipeline.query(question, n_sigma=3.0)   # FAR ≈ 0.13%
```

---

## Running the Demo

```bash
git clone https://github.com/kpal002/sigma-rag
cd sigma-rag
pip install -e ".[dev]"

# Offline demo (no API key)
python demo.py --llm echo

# With Anthropic
ANTHROPIC_API_KEY=sk-... python demo.py --llm anthropic
```

---

## Running Tests

```bash
pytest                        # all tests
pytest -m "not slow"          # skip slow tests
pytest tests/test_retriever.py -v
```

---

## Project Structure

```
sigma-rag/
├── sigma_rag/
│   ├── __init__.py       # public API exports
│   ├── types.py          # Chunk, ScoredChunk, RetrievalResult, RAGResponse
│   ├── stats.py          # pure-numpy norm_cdf, ks_test (scipy optional)
│   ├── noise_floor.py    # NoiseFloor — fits & queries the null distribution
│   ├── embedder.py       # Embedder ABC + SentenceTransformer/OpenAI/Hash backends
│   ├── index.py          # SigmaIndex — document ingestion, chunking, calibration
│   ├── retriever.py      # SigmaRetriever + TopKRetriever baseline
│   └── pipeline.py       # SigmaRAGPipeline — end-to-end QA
├── tests/
│   ├── conftest.py
│   ├── test_embedder.py
│   ├── test_noise_floor.py
│   ├── test_index.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── notebooks/
│   └── demo.ipynb        # σ-RAG vs top-k visual comparison
├── demo.py               # CLI demo script
├── benchmark.py          # benchmark vs top-k
├── pyproject.toml
└── README.md
```

---

## The Physics Backstory

The idea comes from **signal significance testing** in particle physics. When the ATLAS or CMS experiments search for a new particle at the LHC, they don't declare a discovery just because they see "the biggest excess we've found today." They declare a discovery only when the local significance — how many standard deviations above the estimated background the observed excess is — reaches **5σ** (local p-value < 2.87 × 10⁻⁷). Below that bar, the excess is considered consistent with a background fluctuation, and no claim is made.

The procedure has two distinct steps:

1. **Background estimation** — measure the expected yield from known Standard Model processes (QCD multijet, W/Z+jets, top pairs…) using control regions or sidebands in data, *before* looking at the signal region.
2. **Significance gate** — only if the observed excess clears the threshold does the experiment report evidence of a new signal.

Standard RAG lacks both steps. It has no background model and no significance gate — it always returns the top-k chunks regardless of whether any of them are actually relevant. σ-RAG imports the same two-step logic into the retrieval layer: estimate the background distribution of cosine similarities from random document pairs, set a threshold with interpretable false-alarm semantics (default 2σ ≈ 2.3% FAR), and refuse to pass sub-threshold context to the LLM.

---

## Citation

If you use σ-RAG in research, please cite:

```bibtex
@software{pal2025sigmarag,
  author  = {Pal, Kuntal},
  title   = {σ-RAG: Significance-Threshold Retrieval for RAG Pipelines},
  year    = {2025},
  url     = {https://github.com/kpal002/sigma-rag},
}
```

---

## License

MIT © [Kuntal Pal](https://github.com/kpal002)
