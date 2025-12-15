"""Qdrant implementation for vector storage."""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger
import uuid

from .base import VectorDBBase, Document, SearchResult


class QdrantStore(VectorDBBase):
    """Qdrant implementation of vector database."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[Any] = None,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        dimension: int = 768,
        distance: str = "cosine"
    ):
        """
        Initialize Qdrant store.

        Args:
            collection_name: Name of the collection
            embedding_model: Embedding model for generating vectors
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for cloud deployment
            dimension: Dimension of embedding vectors
            distance: Distance metric ('cosine', 'euclidean', 'dot')
        """
        super().__init__(collection_name, embedding_model)

        # Initialize client
        if api_key:
            self.client = QdrantClient(url=f"https://{host}", api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(distance, Distance.COSINE)

        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=self.distance
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Connected to existing Qdrant collection: {collection_name}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Qdrant."""
        try:
            points = []
            ids = []

            for doc in documents:
                doc_id = doc.id or str(uuid.uuid4())
                ids.append(doc_id)

                # Generate embedding
                embedding = doc.embedding or self.embed_text(doc.content)

                # Create point
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc.content,
                        **doc.metadata
                    }
                )
                points.append(point)

            # Upsert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(documents)} documents to Qdrant")
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
        """Search for similar documents in Qdrant."""
        try:
            query_embedding = self.embed_text(query)

            # Build filter
            query_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)

            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter
            )

            search_results = []
            for result in results:
                payload = result.payload.copy()
                content = payload.pop("content", "")

                doc = Document(
                    content=content,
                    metadata=payload,
                    id=str(result.id)
                )

                search_results.append(SearchResult(
                    document=doc,
                    score=result.score
                ))

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Qdrant."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=document_ids
            )
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document in Qdrant."""
        try:
            embedding = document.embedding or self.embed_text(document.content)

            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload={
                    "content": document.content,
                    **document.metadata
                }
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "document_count": info.points_count,
                "vectors_count": info.vectors_count,
                "backend": "Qdrant"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
