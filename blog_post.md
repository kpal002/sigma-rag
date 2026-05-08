---
title: "σ-RAG: What Particle Physics Taught Me About RAG Hallucinations"
date: "2025-01-15"
tags: ["AI", "RAG", "LLM", "NLP", "physics", "open-source"]
summary: "Standard RAG always returns top-k chunks — even when none of them are relevant. I built σ-RAG to fix this using a technique from particle physics: estimate the background distribution, set a significance threshold, and refuse to answer when the evidence isn't there."
---

Last year I was trying to build a RAG system over a mixed corpus of documents when I noticed something uncomfortable: when I asked questions the corpus *couldn't* answer, the LLM still gave confident, detailed responses. Complete hallucinations — dressed up as factual answers.

The culprit was the retriever. It was doing exactly what it was told: return the 3 most similar chunks. But "most similar" doesn't mean *actually similar*. When your query is about pasta carbonara and your corpus contains only physics papers, the top-3 chunks are just the least-irrelevant physics papers — and the LLM has no way to know they're background.

I'd seen this problem before. Just not in NLP.

## The Signal Significance Problem

During my PhD in particle physics, one of the first things you learn is that you don't declare a new particle discovery just because you see "the largest excess we've found today." The ATLAS and CMS experiments at the LHC require a **local significance of 5σ** — meaning the probability that background processes alone could produce an excess at least as large as observed is below 2.87 × 10⁻⁷. Below that threshold, the excess is consistent with a background fluctuation, and no claim is made.

The procedure has two distinct steps that are always kept separate:

1. **Background estimation** — measure the expected yield from known processes in control regions or sidebands *before* looking at the signal region.
2. **Significance gate** — only if the observed excess clears the threshold do you report evidence of a new signal.

When the Higgs boson was discovered in 2012, both ATLAS and CMS independently crossed the 5σ bar in the diphoton and four-lepton channels simultaneously. If either experiment had declared a discovery every time it saw "the best bump this week," it would have announced hundreds of false discoveries over the years.

Standard RAG has neither step. It has no background model and no significance gate — it always returns the top-k chunks regardless of whether any of them are genuinely relevant. And when today's top-k is background, the LLM hallucinates.

The fix seemed obvious: borrow the same two-step logic. Estimate the background. Set an absolute threshold. Refuse to answer when the evidence doesn't clear the bar.

## How σ-RAG Works

The key insight is that cosine similarities between *random, unrelated document pairs* form a distribution — the background. If you sample enough random cross-document pairs from your corpus, you get a Gaussian with some mean μ and standard deviation σ. This is the baseline: what a background-level match looks like in your particular embedding space.

At query time, instead of asking "which chunks are most similar?", σ-RAG asks "which chunks are significantly above background?" Specifically:

```
threshold = μ_background + n·σ_background   (default n = 2)
```

A chunk clears the bar only if its similarity exceeds this threshold. At n=2, the false-alarm rate — the probability that a genuinely irrelevant chunk exceeds the threshold — is about 2.3%. At n=3 it drops to 0.13%. At n=5 it hits 2.9 × 10⁻⁷, the LHC discovery bar.

If zero chunks clear the bar, the pipeline returns a calibrated "no evidence" response and **never calls the LLM**. No background context → no hallucination opportunity.

```python
from sigma_rag import SigmaIndex, SigmaRAGPipeline

index = SigmaIndex()
index.add_documents(corpus_docs)
index.calibrate()   # estimates the background from random cross-document pairs

pipeline = SigmaRAGPipeline(index, n_sigma=2.0)

# Answerable query: retrieves significant chunks, answers normally
response = pipeline.query("What significance level was required for the Higgs discovery?")
print(response.has_evidence)   # True
print(f"Used {len(response.retrieval.significant)} chunks")

# Unanswerable query: suppressed entirely
response = pipeline.query("What is the best carbonara recipe?")
print(response.has_evidence)   # False  ← hallucination prevented
```

## The Results

I tested on a mixed corpus of particle physics papers and cooking articles, with a set of answerable (physics) and unanswerable (cooking) queries:

| Metric | Top-k (k=3) | σ-RAG (2σ) |
|--------|:-----------:|:----------:|
| Precision on answerable queries | 100% | 100% |
| **Hallucination risk on unanswerable** | **100%** | **0%** |
| Avg. chunks passed to LLM | 3.0 | 1.8 |

σ-RAG matches standard RAG on answerable questions. The difference shows up precisely where it matters: queries the system genuinely can't answer.

## Implementation Notes

A few things I found interesting to build:

**Making it dependency-free.** I wanted the package to run with only `numpy` — no `scipy`, no `sentence-transformers`, no API keys needed. This meant writing a pure-numpy normal CDF via the `math.erfc` identity, a Kolmogorov-Smirnov test approximation, and a hash-based bag-of-words embedder as a fallback. The package auto-detects scipy and sentence-transformers when available and upgrades seamlessly.

**Cross-document sampling.** Sampling pairs from the *same* document would produce artificially high cosine similarities (related chunks from the same text are more similar than random). I use cross-document pairs only, which gives a cleaner background estimate — analogous to measuring the background in a sideband rather than in the signal region itself.

**The KS test warning.** After fitting, I run a KS test on the sampled similarities against the Gaussian fit. If the distribution is significantly non-Gaussian (which happens with hash embedders on small corpora), the package logs a warning. This is a useful signal — it means the threshold's false-alarm semantics are approximate, and you should switch to sentence-transformers for production use.

**Adaptive thresholds.** Sometimes you want at least one result even if nothing is clearly significant. `SigmaRetriever` supports a `min_results` parameter that relaxes the threshold progressively if too few chunks pass, down to a configurable `min_sigma` floor.

## What's Next

The most interesting open question is whether to learn the background rather than estimate it empirically. For very small corpora, the Gaussian approximation degrades — a kernel density estimate or normalising flow might model the null distribution more accurately. This is essentially the same trade-off between parametric and non-parametric background modelling that occupies a lot of time in HEP analyses.

There's also the question of per-query calibration: the background varies across different types of queries (domain-specific vs. general), and a query-conditional threshold might be more principled than a global one — similar to how particle physics analyses use different background models in different kinematic regions.

The code is on GitHub: [github.com/kpal002/sigma-rag](https://github.com/kpal002/sigma-rag). Install with `pip install "sigma-rag[local]"` for the full local embedding setup, or just `pip install sigma-rag` for the zero-dependency version. The demo notebook walks through all of this with plots.

---

*Kuntal Pal is an AI Engineer and Particle Physics PhD. He works on agentic AI systems and occasionally lets his HEP background bleed into production code.*
