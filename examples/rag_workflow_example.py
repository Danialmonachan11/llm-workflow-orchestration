"""Example of Retrieval-Augmented Generation (RAG) workflow."""

import asyncio
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.agents import (
    LLMAgent,
    RetrievalAgent,
    AgentOrchestrator,
    AgentConfig,
    AgentRole,
    WorkflowConfig,
    WorkflowStep,
    WorkflowType
)
from src.vectordb import ChromaDBStore, Document
from src.utils import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(log_level="INFO")


async def main():
    """Run RAG workflow example."""

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize vector database
    print("Initializing vector database...")
    vector_db = ChromaDBStore(
        collection_name="knowledge_base",
        embedding_model=embedding_model,
        persist_directory="./data/rag_example"
    )

    # Add knowledge base documents
    print("Adding documents to knowledge base...")
    knowledge_docs = [
        Document(
            content="Transformers are a type of neural network architecture that uses self-attention mechanisms. They were introduced in the 'Attention is All You Need' paper in 2017.",
            metadata={"topic": "transformers", "type": "architecture"}
        ),
        Document(
            content="BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed for natural language understanding tasks. It uses masked language modeling for pre-training.",
            metadata={"topic": "transformers", "type": "model"}
        ),
        Document(
            content="GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates text by predicting the next token. It uses decoder-only transformer architecture.",
            metadata={"topic": "transformers", "type": "model"}
        ),
        Document(
            content="Self-attention allows transformers to weigh the importance of different words in a sequence when processing each word. This enables better context understanding.",
            metadata={"topic": "transformers", "type": "mechanism"}
        ),
        Document(
            content="Vision Transformers (ViT) apply transformer architecture to image classification by treating image patches as tokens, similar to words in NLP.",
            metadata={"topic": "transformers", "type": "application"}
        ),
    ]
    vector_db.add_documents(knowledge_docs)

    # Create Retrieval Agent
    retrieval_config = AgentConfig(
        name="retriever",
        role=AgentRole.RETRIEVER,
        system_prompt="Retrieve relevant information from the knowledge base."
    )

    retrieval_agent = RetrievalAgent(
        config=retrieval_config,
        vector_db=vector_db,
        top_k=3,
        similarity_threshold=0.3
    )

    # Create LLM Agent for generation
    generation_config = AgentConfig(
        name="generator",
        role=AgentRole.GENERATOR,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2000,
        system_prompt="""You are an AI assistant that provides accurate answers based on retrieved information.
Use the retrieved context to answer questions comprehensively.
If the context doesn't contain enough information, acknowledge this."""
    )

    generation_agent = LLMAgent(
        config=generation_config,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    # Create orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(retrieval_agent)
    orchestrator.register_agent(generation_agent)

    # Define RAG workflow
    workflow = WorkflowConfig(
        name="rag_qa",
        workflow_type=WorkflowType.RAG,
        steps=[
            WorkflowStep(
                agent_name="retriever",
                action="retrieve"
            ),
            WorkflowStep(
                agent_name="generator",
                action="generate"
            )
        ],
        initial_step="retriever"
    )

    orchestrator.register_workflow(workflow)

    # Execute RAG workflow with queries
    queries = [
        "What is a transformer in machine learning?",
        "How does BERT differ from GPT?",
        "Can transformers be used for computer vision?"
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"\nQuery: {query}")
        print("-" * 80)

        result = await orchestrator.execute_workflow(
            workflow_name="rag_qa",
            initial_input=query
        )

        if result.success:
            print("\nRetrieved Information:")
            print(result.outputs.get("retrieval", "No retrieval results"))
            print("\n" + "-" * 80)
            print("\nGenerated Answer:")
            print(result.outputs.get("generation", "No generation results"))
        else:
            print(f"\nError: {result.error}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
