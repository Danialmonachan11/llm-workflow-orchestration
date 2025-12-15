"""Base class for vector database implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation for vector storage."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    id: Optional[str] = None


@dataclass
class SearchResult:
    """Search result from vector database."""

    document: Document
    score: float
    distance: Optional[float] = None


class VectorDBBase(ABC):
    """Abstract base class for vector database implementations."""

    def __init__(self, collection_name: str, embedding_model: Optional[Any] = None):
        """
        Initialize vector database.

        Args:
            collection_name: Name of the collection/index
            embedding_model: Model to generate embeddings (optional)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the database.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document: Document) -> bool:
        """
        Update a document in the database.

        Args:
            document_id: ID of document to update
            document: Updated document

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary containing collection statistics
        """
        pass

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            raise ValueError("No embedding model configured")

        if hasattr(self.embedding_model, 'encode'):
            return self.embedding_model.encode(text).tolist()
        else:
            raise NotImplementedError("Embedding model must have 'encode' method")
