"""Shared RAG-stack bootstrap: embedding model, vector store(s), agents,
and orchestrator/workflow wiring.

Used by the FastAPI service (`app.py`, same package), the RAGAS evaluation
script (`src/evaluation/ragas_eval.py`), and `examples/rag_workflow_example.py`
-- one place this setup logic lives, instead of three copies of it.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Loaded here, not left to each caller, so every entry point that uses this
# module (the FastAPI service under plain uvicorn, the RAGAS eval script,
# examples/rag_workflow_example.py) reliably gets OPENROUTER_API_KEY/etc
# from .env regardless of whether it remembered to call load_dotenv() itself.
load_dotenv()

from ..agents import (
    LLMAgent,
    RetrievalAgent,
    AgentOrchestrator,
    AgentConfig,
    AgentRole,
    WorkflowConfig,
    WorkflowStep,
    WorkflowType,
)
from ..vectordb import ChromaDBStore, Document
from ..vectordb.qdrant_db import QdrantStore

# Same 5-document transformer-topic knowledge base used throughout this
# repo's RAG examples/tests, so results are directly comparable across the
# example script, the API, and the RAGAS eval's retrieval step.
KNOWLEDGE_DOCS = [
    Document(
        content="Transformers are a type of neural network architecture that uses self-attention mechanisms. They were introduced in the 'Attention is All You Need' paper in 2017.",
        metadata={"topic": "transformers", "type": "architecture"},
    ),
    Document(
        content="BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed for natural language understanding tasks. It uses masked language modeling for pre-training.",
        metadata={"topic": "transformers", "type": "model"},
    ),
    Document(
        content="GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates text by predicting the next token. It uses decoder-only transformer architecture.",
        metadata={"topic": "transformers", "type": "model"},
    ),
    Document(
        content="Self-attention allows transformers to weigh the importance of different words in a sequence when processing each word. This enables better context understanding.",
        metadata={"topic": "transformers", "type": "mechanism"},
    ),
    Document(
        content="Vision Transformers (ViT) apply transformer architecture to image classification by treating image patches as tokens, similar to words in NLP.",
        metadata={"topic": "transformers", "type": "application"},
    ),
]


def _resolve_llm_settings() -> tuple[str, Optional[str], str]:
    """Pick an API key/base_url/model triple.

    Prefers OpenRouter (`OPENROUTER_API_KEY`) so this repo can reuse the
    same key already proven working in the sibling finance-agent project --
    one OpenAI-compatible client, any OpenRouter-hosted model, no separate
    provider signup needed. Falls back to a native OPENAI_API_KEY if no
    OpenRouter key is set.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        model_name = os.getenv("RAG_MODEL", "anthropic/claude-haiku-4.5")
        return openrouter_key, "https://openrouter.ai/api/v1", model_name

    openai_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("RAG_MODEL", "gpt-3.5-turbo")
    return openai_key, None, model_name


def build_orchestrator(
    persist_directory: str = "./data/rag_store",
    use_hybrid: bool = True,
    use_reranking: bool = True,
    use_qdrant: bool = False,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> AgentOrchestrator:
    """Build and return an AgentOrchestrator with a registered 'rag_qa'
    workflow, ready to call `execute_workflow('rag_qa', query)` on.

    Args:
        persist_directory: Chroma's local persistence directory.
        use_hybrid: build a BM25 keyword index over KNOWLEDGE_DOCS and
            fuse it with dense search (hybrid dense+sparse retrieval).
        use_reranking: rerank fused candidates with a cross-encoder.
        use_qdrant: also stand up a Qdrant-backed secondary vector store
            and fuse its results with the primary Chroma store (the
            actual "combining Chroma and Qdrant" behaviour). Requires a
            reachable Qdrant server (`docker compose up qdrant`).
    """
    api_key, base_url, model_name = _resolve_llm_settings()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_db = ChromaDBStore(
        collection_name="knowledge_base",
        embedding_model=embedding_model,
        persist_directory=persist_directory,
    )
    if vector_db.get_collection_stats().get("document_count", 0) == 0:
        vector_db.add_documents(KNOWLEDGE_DOCS)

    secondary_db = None
    if use_qdrant:
        secondary_db = QdrantStore(
            collection_name="knowledge_base",
            embedding_model=embedding_model,
            host=qdrant_host,
            port=qdrant_port,
            dimension=embedding_model.get_sentence_embedding_dimension(),
        )
        if secondary_db.get_collection_stats().get("document_count", 0) == 0:
            secondary_db.add_documents(KNOWLEDGE_DOCS)

    retrieval_config = AgentConfig(
        name="retriever",
        role=AgentRole.RETRIEVER,
        system_prompt="Retrieve relevant information from the knowledge base.",
    )
    retrieval_agent = RetrievalAgent(
        config=retrieval_config,
        vector_db=vector_db,
        top_k=3,
        similarity_threshold=0.3,
        secondary_vector_db=secondary_db,
        bm25_corpus=KNOWLEDGE_DOCS if use_hybrid else None,
        use_reranking=use_reranking,
    )

    generation_config = AgentConfig(
        name="generator",
        role=AgentRole.GENERATOR,
        model_name=model_name,
        temperature=0.7,
        max_tokens=2000,
        system_prompt=(
            "You are an AI assistant that provides accurate answers based on retrieved information.\n"
            "Use the retrieved context to answer questions comprehensively.\n"
            "If the context doesn't contain enough information, acknowledge this."
        ),
    )
    generation_agent = LLMAgent(
        config=generation_config,
        api_key=api_key,
        provider="openai",
        base_url=base_url,
    )

    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(retrieval_agent)
    orchestrator.register_agent(generation_agent)

    workflow = WorkflowConfig(
        name="rag_qa",
        workflow_type=WorkflowType.RAG,
        steps=[
            WorkflowStep(agent_name="retriever", action="retrieve"),
            WorkflowStep(agent_name="generator", action="generate"),
        ],
        initial_step="retriever",
    )
    orchestrator.register_workflow(workflow)

    return orchestrator
