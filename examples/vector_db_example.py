"""Example of using vector databases for document storage and retrieval."""

import asyncio
from sentence_transformers import SentenceTransformer

from src.vectordb import ChromaDBStore, Document
from src.utils import setup_logging

# Setup logging
setup_logging(log_level="INFO")


def main():
    """Run vector database example."""

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    vector_db = ChromaDBStore(
        collection_name="example_documents",
        embedding_model=embedding_model,
        persist_directory="./data/chroma_example"
    )

    # Create sample documents
    documents = [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"topic": "programming", "language": "python"}
        ),
        Document(
            content="Machine learning is a subset of AI that enables systems to learn from data.",
            metadata={"topic": "ai", "category": "machine_learning"}
        ),
        Document(
            content="Neural networks are computing systems inspired by biological neural networks.",
            metadata={"topic": "ai", "category": "deep_learning"}
        ),
        Document(
            content="JavaScript is a versatile programming language primarily used for web development.",
            metadata={"topic": "programming", "language": "javascript"}
        ),
        Document(
            content="Natural Language Processing enables computers to understand human language.",
            metadata={"topic": "ai", "category": "nlp"}
        ),
    ]

    # Add documents to vector database
    print("\nAdding documents to vector database...")
    doc_ids = vector_db.add_documents(documents)
    print(f"Added {len(doc_ids)} documents")

    # Search for similar documents
    print("\n" + "=" * 80)
    query1 = "Tell me about programming languages"
    print(f"\nQuery 1: {query1}")
    results1 = vector_db.search(query=query1, top_k=3)

    print("\nResults:")
    for i, result in enumerate(results1, 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Content: {result.document.content}")
        print(f"   Metadata: {result.document.metadata}")

    # Search with metadata filter
    print("\n" + "=" * 80)
    query2 = "artificial intelligence concepts"
    print(f"\nQuery 2: {query2}")
    print("Filter: topic='ai'")
    results2 = vector_db.search(
        query=query2,
        top_k=3,
        filter_metadata={"topic": "ai"}
    )

    print("\nResults:")
    for i, result in enumerate(results2, 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Content: {result.document.content}")
        print(f"   Metadata: {result.document.metadata}")

    # Get collection stats
    print("\n" + "=" * 80)
    print("\nCollection Statistics:")
    stats = vector_db.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
