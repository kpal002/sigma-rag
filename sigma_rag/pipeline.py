"""
sigma_rag/pipeline.py
---------------------
End-to-end σ-RAG pipeline: retrieve → generate → respond.

SigmaRAGPipeline wraps SigmaRetriever + an LLM to provide a complete
question-answering interface.  The key behaviour that distinguishes it
from a standard RAG pipeline:

  - If no chunk clears the significance threshold, the pipeline returns
    a calibrated "insufficient evidence" response rather than asking the
    LLM to hallucinate from noise-level context.

  - The final answer includes metadata about how many chunks were used,
    their z-scores, and the noise floor statistics — so the caller can
    make informed decisions about confidence.

Supported LLM backends:
  - Anthropic (claude-* models)   — primary
  - OpenAI   (gpt-* models)       — secondary
  - Echo     (no API key needed)  — for testing / offline use
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from sigma_rag.index import SigmaIndex
from sigma_rag.retriever import SigmaRetriever, TopKRetriever
from sigma_rag.types import RAGResponse, RetrievalResult

logger = logging.getLogger(__name__)

# Default system prompt — instructs the LLM to answer only from context
_SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information to answer the question,
say exactly: "The provided context does not contain enough information to answer this question."
Do not invent information. Be concise and factual."""

_NO_EVIDENCE_ANSWER = (
    "⚠️  σ-RAG: No significant evidence found in the corpus for this query.\n"
    "All retrieved chunks fell below the significance threshold "
    "(z < {n_sigma}σ, FAR ≈ {far:.1%}). "
    "This response is suppressed to prevent hallucination."
)


class SigmaRAGPipeline:
    """
    Full question-answering pipeline with significance-gated retrieval.

    Args:
        index:       A calibrated SigmaIndex.
        n_sigma:     Significance threshold for retrieval.
        max_results: Max chunks to pass to the LLM as context.
        llm:         LLM backend: 'anthropic', 'openai', or 'echo'.
        model:       Model name.  Defaults to claude-haiku-4-5 / gpt-4o-mini.
        system_prompt: Override the default system prompt.
        temperature: LLM temperature.  Keep low (0.0–0.3) for factual QA.

    Example (offline / no API key):
        >>> index = SigmaIndex()
        >>> index.add_documents(docs).calibrate()
        >>> pipeline = SigmaRAGPipeline(index, llm="echo")
        >>> response = pipeline.query("What is dark matter?")
        >>> print(response.answer)

    Example (Anthropic):
        >>> import os; os.environ["ANTHROPIC_API_KEY"] = "sk-..."
        >>> pipeline = SigmaRAGPipeline(index, llm="anthropic")
        >>> response = pipeline.query("What is dark matter?")
    """

    def __init__(
        self,
        index: SigmaIndex,
        n_sigma: float | None = None,
        max_results: int = 5,
        llm: Literal["anthropic", "openai", "echo"] = "anthropic",
        model: str | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
        temperature: float = 0.1,
    ) -> None:
        self.index = index
        self.n_sigma = n_sigma if n_sigma is not None else index.n_sigma
        self.max_results = max_results
        self.llm_backend = llm
        self.temperature = temperature
        self.system_prompt = system_prompt

        # Instantiate retriever
        self.retriever = SigmaRetriever(
            index=index,
            n_sigma=self.n_sigma,
            max_results=max_results,
        )

        # Set model default per backend
        if model is not None:
            self.model = model
        elif llm == "anthropic":
            self.model = "claude-haiku-4-5-20251001"
        elif llm == "openai":
            self.model = "gpt-4o-mini"
        else:
            self.model = "echo"

        # Eagerly instantiate LLM clients (fail fast, not on first query)
        self._llm_client: object = None
        if llm == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                logger.warning(
                    "ANTHROPIC_API_KEY not set. Calls to .query() will fail. "
                    "Use llm='echo' for offline testing."
                )
            else:
                try:
                    import anthropic as _anthropic

                    self._llm_client = _anthropic.Anthropic()
                except ImportError:
                    logger.warning(
                        "anthropic package not installed. Install with: pip install anthropic"
                    )
        elif llm == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not set.")
            else:
                try:
                    from openai import OpenAI as _OpenAI

                    self._llm_client = _OpenAI()
                except ImportError:
                    logger.warning("openai package not installed. Install with: pip install openai")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, n_sigma: float | None = None) -> RAGResponse:
        """
        Answer a question using significance-gated RAG.

        Args:
            question: Natural language question.
            n_sigma:  Per-call override for significance threshold.

        Returns:
            RAGResponse with answer, retrieval stats, and evidence flag.
        """
        self.index.check_ready()

        # ── Retrieve ──────────────────────────────────────────────────
        result: RetrievalResult = self.retriever.retrieve(question, n_sigma=n_sigma)

        # ── Gate: no evidence → suppress generation ───────────────────
        if not result.has_evidence:
            from sigma_rag import stats as _stats

            far = _stats.sf(result.n_sigma)
            answer = _NO_EVIDENCE_ANSWER.format(n_sigma=result.n_sigma, far=far)
            logger.info(
                "No significant evidence for query %r (threshold=%.4f, n_sigma=%.1f)",
                question[:60],
                result.threshold,
                result.n_sigma,
            )
            return RAGResponse(
                answer=answer,
                retrieval=result,
                has_evidence=False,
                model=self.model,
                context_used="",
            )

        # ── Build context from significant chunks ─────────────────────
        context = self._build_context(result)

        # ── Generate ──────────────────────────────────────────────────
        answer = self._generate(question, context)

        return RAGResponse(
            answer=answer,
            retrieval=result,
            has_evidence=True,
            model=self.model,
            context_used=context,
        )

    def compare_with_topk(self, question: str, k: int = 5) -> dict:
        """
        Run both σ-RAG and standard top-k on the same question.

        Useful for side-by-side demonstration of the difference.

        Args:
            question: The query string.
            k:        k for the top-k baseline.

        Returns:
            Dict with keys 'sigma_rag' and 'top_k', each a RAGResponse.
        """
        sigma_response = self.query(question)

        # Top-k retrieval
        topk_retriever = TopKRetriever(self.index, k=k)
        topk_result = topk_retriever.retrieve(question)
        topk_context = self._build_context(topk_result)
        topk_answer = self._generate(question, topk_context)
        topk_response = RAGResponse(
            answer=topk_answer,
            retrieval=topk_result,
            has_evidence=True,
            model=self.model,
            context_used=topk_context,
        )

        return {"sigma_rag": sigma_response, "top_k": topk_response}

    # ------------------------------------------------------------------
    # Private: context building
    # ------------------------------------------------------------------

    def _build_context(self, result: RetrievalResult) -> str:
        """
        Format significant chunks into a numbered context string for the LLM.

        Args:
            result: A RetrievalResult with at least one significant chunk.

        Returns:
            Formatted context string.
        """
        lines = [
            f"[Noise floor: μ={result.noise_mu:.4f}, σ={result.noise_sigma:.4f}, "
            f"threshold={result.threshold:.4f} @ {result.n_sigma}σ]\n"
        ]
        for i, sc in enumerate(result.significant, start=1):
            lines.append(
                f"[Passage {i} | similarity={sc.similarity:.4f} | "
                f"z={sc.z_score:.2f}σ | p={sc.p_value:.4f}]\n"
                f"{sc.chunk.text}\n"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private: LLM generation
    # ------------------------------------------------------------------

    def _generate(self, question: str, context: str) -> str:
        """
        Call the configured LLM to answer the question given context.

        Args:
            question: User question.
            context:  Formatted context string from significant chunks.

        Returns:
            Generated answer string.
        """
        if self.llm_backend == "anthropic":
            return self._generate_anthropic(question, context)
        elif self.llm_backend == "openai":
            return self._generate_openai(question, context)
        elif self.llm_backend == "echo":
            return self._generate_echo(question, context)
        else:
            raise ValueError(f"Unknown LLM backend: {self.llm_backend!r}")

    def _generate_anthropic(self, question: str, context: str) -> str:
        """Generate using the Anthropic Messages API."""
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from exc

        if self._llm_client is None:
            self._llm_client = _anthropic.Anthropic()

        client = self._llm_client
        assert isinstance(client, _anthropic.Anthropic)

        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based only on the context above:"
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        block = response.content[0]
        assert isinstance(block, _anthropic.types.TextBlock)
        return str(block.text).strip()

    def _generate_openai(self, question: str, context: str) -> str:
        """Generate using the OpenAI Chat Completions API."""
        try:
            from openai import OpenAI as _OpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        if self._llm_client is None:
            self._llm_client = _OpenAI()

        client = self._llm_client
        assert isinstance(client, _OpenAI)

        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based only on the context above:"
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content is not None else ""

    def _generate_echo(self, question: str, context: str) -> str:
        """Echo backend — returns context as-is (no LLM call).

        Used for offline testing and benchmarking retrieval quality
        independently of LLM quality.
        """
        return f"[ECHO] Question: {question}\n[ECHO] Context passages used:\n{context}"
