"""Langflow component for vector database operations."""

from langflow import CustomComponent
from typing import Optional, Dict, Any, List
from langchain.schema import Document

from ..vectordb import ChromaDBStore, FAISSStore, PineconeStore, QdrantStore
from ..vectordb.base import Document as VectorDocument


class VectorDBComponent(CustomComponent):
    """Custom Langflow component for vector database operations."""

    display_name = "Vector Database"
    description = "Store and retrieve documents from vector databases"
    documentation = "https://github.com/yourusername/llm-workflow-orchestration"

    def build_config(self) -> Dict[str, Any]:
        """Build component configuration."""
        return {
            "db_type": {
                "display_name": "Database Type",
                "options": ["ChromaDB", "FAISS", "Pinecone", "Qdrant"],
                "value": "ChromaDB",
            },
            "collection_name": {
                "display_name": "Collection Name",
                "value": "default_collection",
            },
            "operation": {
                "display_name": "Operation",
                "options": ["search", "add", "delete"],
                "value": "search",
            },
            "query": {
                "display_name": "Query",
                "multiline": True,
                "value": "",
            },
            "documents": {
                "display_name": "Documents to Add",
                "multiline": True,
                "value": "",
            },
            "top_k": {
                "display_name": "Top K Results",
                "value": 5,
            },
            "persist_directory": {
                "display_name": "Persist Directory",
                "value": "./vector_db",
            },
            "api_key": {
                "display_name": "API Key (for cloud DBs)",
                "password": True,
                "value": "",
            },
        }

    def build(
        self,
        db_type: str,
        collection_name: str,
        operation: str,
        query: str = "",
        documents: str = "",
        top_k: int = 5,
        persist_directory: str = "./vector_db",
        api_key: str = "",
    ) -> Dict[str, Any]:
        """
        Execute vector database operation.

        Args:
            db_type: Type of vector database
            collection_name: Name of collection/index
            operation: Operation to perform
            query: Search query
            documents: Documents to add (newline separated)
            top_k: Number of results
            persist_directory: Directory to persist data
            api_key: API key for cloud databases

        Returns:
            Operation result
        """
        # Initialize vector database
        vector_db = self._init_db(
            db_type=db_type,
            collection_name=collection_name,
            persist_directory=persist_directory,
            api_key=api_key
        )

        # Execute operation
        if operation == "search":
            results = vector_db.search(query=query, top_k=top_k)
            return {
                "results": [
                    {
                        "content": r.document.content,
                        "score": r.score,
                        "metadata": r.document.metadata
                    }
                    for r in results
                ],
                "num_results": len(results)
            }

        elif operation == "add":
            # Parse documents
            doc_list = []
            for i, doc_text in enumerate(documents.split("\n")):
                if doc_text.strip():
                    doc_list.append(VectorDocument(
                        content=doc_text.strip(),
                        metadata={"source": "langflow"}
                    ))

            # Add documents
            ids = vector_db.add_documents(doc_list)
            return {
                "status": "success",
                "num_added": len(ids),
                "ids": ids
            }

        elif operation == "delete":
            # Query should contain document IDs (comma separated)
            doc_ids = [id.strip() for id in query.split(",")]
            success = vector_db.delete_documents(doc_ids)
            return {
                "status": "success" if success else "failed",
                "num_deleted": len(doc_ids)
            }

        else:
            return {"error": f"Unknown operation: {operation}"}

    def _init_db(
        self,
        db_type: str,
        collection_name: str,
        persist_directory: str,
        api_key: str
    ):
        """Initialize vector database based on type."""
        # Note: In real implementation, you'd need to provide embedding model
        # This is a simplified version

        if db_type == "ChromaDB":
            return ChromaDBStore(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        elif db_type == "FAISS":
            return FAISSStore(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        elif db_type == "Pinecone":
            # Parse API key and environment from api_key field
            # Format: "api_key:environment"
            parts = api_key.split(":")
            key = parts[0] if parts else ""
            env = parts[1] if len(parts) > 1 else "us-west1-gcp"

            return PineconeStore(
                collection_name=collection_name,
                api_key=key,
                environment=env
            )
        elif db_type == "Qdrant":
            return QdrantStore(
                collection_name=collection_name
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
