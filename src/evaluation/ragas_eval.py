"""Real RAGAS evaluation of the RAG workflow -- runs the actual pipeline
(hybrid retrieval + reranking + generation, via the same orchestrator the
API and examples use) against a small golden question set, then scores it
with RAGAS's faithfulness / answer_relevancy / context_precision /
context_recall metrics.

This is a genuine evaluation run, not a placeholder: every question below
goes through real retrieval and a real LLM call, and the printed scores
come from RAGAS actually computing them against those real outputs.

Run: python -m src.evaluation.ragas_eval
"""

import asyncio
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

# ragas.llms.base unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai to register it as one of many
# optional supported LLM providers -- but that submodule was removed from
# langchain-community during its 2026 "sunset" deprecation, while every
# other package in this repo's stack (langgraph, langchain-openai) needs
# the langchain-community version that removed it. We never use Vertex AI
# here; this stub satisfies ragas's eager import so the rest of the module
# (which we do use) loads. Verified: only this one Vertex AI class is
# actually missing (langchain_community.llms.VertexAI, imported on the
# next line in ragas' source, still resolves fine on its own).
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:  # pragma: no cover - never instantiated, import-satisfying only
        pass

    _vertexai_stub.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from src.api.bootstrap import _resolve_llm_settings, build_orchestrator

# Golden question / ground-truth-answer pairs, each answerable from the
# 5-document knowledge base in bootstrap.KNOWLEDGE_DOCS.
GOLDEN_QA = [
    ("What paper introduced the transformer architecture?", "The 'Attention is All You Need' paper, published in 2017."),
    ("What does BERT stand for?", "Bidirectional Encoder Representations from Transformers."),
    ("What pre-training technique does BERT use?", "Masked language modeling."),
    ("What does GPT stand for?", "Generative Pre-trained Transformer."),
    ("What kind of transformer architecture does GPT use?", "A decoder-only transformer architecture."),
    ("What mechanism allows transformers to weigh the importance of different words in a sequence?", "Self-attention."),
    ("How do Vision Transformers process images?", "By treating image patches as tokens, similar to words in NLP."),
    ("Is GPT an autoregressive or a bidirectional model?", "GPT is an autoregressive language model."),
]


def _split_retrieved_contexts(retrieved_text: str) -> list[str]:
    """RAGAS wants `contexts` as a list of separate context strings.
    RetrievalAgent formats results as a numbered "N. [Score: x.xx]\\n   content"
    block; split back into individual context strings."""
    chunks = re.split(r"\n?\d+\. \[Score: [\d.]+\]\n?\s*", retrieved_text)
    return [c.strip() for c in chunks if c.strip() and c.strip() != "No relevant information found."]


async def run_eval() -> None:
    orchestrator = build_orchestrator(use_hybrid=True, use_reranking=True)

    questions, answers, contexts, ground_truths = [], [], [], []

    for question, ground_truth in GOLDEN_QA:
        result = await orchestrator.execute_workflow(workflow_name="rag_qa", initial_input=question)
        if not result.success:
            print(f"[SKIP] '{question}' -> workflow failed: {result.error}")
            continue

        questions.append(question)
        answers.append(result.outputs.get("generation", ""))
        contexts.append(_split_retrieved_contexts(result.outputs.get("retrieval", "")) or [""])
        ground_truths.append(ground_truth)
        print(f"[ran] {question}")

    if not questions:
        print("ragas_eval: no successful workflow runs to evaluate -- check API key/config.")
        return

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # RAGAS's metrics need a "judge" LLM/embeddings of their own to score
    # the real outputs above -- by default it reaches for a raw OpenAI()
    # client via OPENAI_API_KEY, which this repo doesn't use. Wrap our own
    # OpenRouter-configured ChatOpenAI (same key/model resolution as the
    # RAG pipeline itself) and a local HuggingFace embedding model instead,
    # so the judge calls go through the same provider as everything else.
    api_key, base_url, model_name = _resolve_llm_settings()
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0, max_tokens=1024))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    print(f"\nEvaluating {len(questions)} real RAG runs with RAGAS...")
    # max_workers capped low: OpenRouter enforces an in-flight concurrent-
    # request budget tied to account balance: 4 metrics x 8 questions = 32
    # judge-LLM calls fired near-simultaneously at the library's default
    # concurrency (16) tripped '402 in_flight_budget_exhausted' on several
    # jobs. A modest worker cap keeps this real free-tier account under
    # that ceiling instead of needing to raise it with paid credits.
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=4),
    )

    print("\n=== RAGAS scores (real run, not placeholders) ===")
    scores_df = result.to_pandas()
    for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if metric_name in scores_df.columns:
            mean_score = scores_df[metric_name].mean(skipna=True)
            print(f"  {metric_name}: {mean_score:.4f}")
        else:
            print(f"  {metric_name}: column not present in results")


def main() -> None:
    asyncio.run(run_eval())


if __name__ == "__main__":
    main()
