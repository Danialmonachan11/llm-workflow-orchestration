"""Example of Retrieval-Augmented Generation (RAG) workflow.

Setup (embedding model, vector store, agents, orchestrator) now lives in
`src/api/bootstrap.py`, shared with the FastAPI service and the RAGAS
evaluation script -- this example just calls it and runs a few sample
queries through the resulting LangGraph-backed 'rag_qa' workflow.
"""

import asyncio

from dotenv import load_dotenv

from src.api.bootstrap import build_orchestrator
from src.utils import setup_logging

load_dotenv()
setup_logging(log_level="INFO")


async def main():
    """Run RAG workflow example."""

    print("Building RAG stack (embedding model, vector store, agents)...")
    orchestrator = build_orchestrator(persist_directory="./data/rag_example")

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
