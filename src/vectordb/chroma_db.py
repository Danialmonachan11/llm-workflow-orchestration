"""ChromaDB implementation for vector storage."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from loguru import logger

from .base import VectorDBBase, Document, SearchResult


class ChromaDBStore(VectorDBBase):
    """ChromaDB implementation of vector database."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[Any] = None,
        persist_directory: str = "./chroma_db",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize ChromaDB store.

        Args:
            collection_name: Name of the collection
            embedding_model: Embedding model for generating vectors
            persist_directory: Directory to persist data (local mode)
            host: ChromaDB server host (client mode)
            port: ChromaDB server port (client mode)
        """
        super().__init__(collection_name, embedding_model)

        # Initialize client
        if host and port:
            self.client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"Connected to ChromaDB at {host}:{port}")
        else:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized local ChromaDB at {persist_directory}")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "LLM workflow orchestration vector store"}
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB."""
        try:
            ids = [doc.id or f"doc_{i}" for i, doc in enumerate(documents)]
            contents = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Generate embeddings if not provided
            embeddings = []
            for doc in documents:
                if doc.embedding:
                    embeddings.append(doc.embedding)
                else:
                    embeddings.append(self.embed_text(doc.content))

            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in ChromaDB."""
        try:
            query_embedding = self.embed_text(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )

            search_results = []
            for i in range(len(results['ids'][0])):
                doc = Document(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    id=results['ids'][0][i]
                )
                search_results.append(SearchResult(
                    document=doc,
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    distance=results['distances'][0][i]
                ))

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document in ChromaDB."""
        try:
            embedding = document.embedding or self.embed_text(document.content)

            self.collection.update(
                ids=[document_id],
                documents=[document.content],
                metadatas=[document.metadata],
                embeddings=[embedding]
            )

            logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "backend": "ChromaDB"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
