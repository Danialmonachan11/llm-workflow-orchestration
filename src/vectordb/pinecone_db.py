"""Pinecone implementation for vector storage."""

from typing import List, Dict, Any, Optional
import pinecone
from loguru import logger

from .base import VectorDBBase, Document, SearchResult


class PineconeStore(VectorDBBase):
    """Pinecone implementation of vector database."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[Any] = None,
        api_key: str = "",
        environment: str = "",
        dimension: int = 768,
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone store.

        Args:
            collection_name: Name of the index
            embedding_model: Embedding model for generating vectors
            api_key: Pinecone API key
            environment: Pinecone environment
            dimension: Dimension of embedding vectors
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
        """
        super().__init__(collection_name, embedding_model)

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Create or connect to index
        if collection_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric=metric
            )
            logger.info(f"Created Pinecone index: {collection_name}")
        else:
            logger.info(f"Connected to existing Pinecone index: {collection_name}")

        self.index = pinecone.Index(collection_name)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Pinecone."""
        try:
            vectors = []
            ids = []

            for i, doc in enumerate(documents):
                doc_id = doc.id or f"doc_{i}"
                ids.append(doc_id)

                # Generate embedding
                embedding = doc.embedding or self.embed_text(doc.content)

                # Prepare vector with metadata
                vector = (
                    doc_id,
                    embedding,
                    {
                        **doc.metadata,
                        "content": doc.content
                    }
                )
                vectors.append(vector)

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)

            logger.info(f"Added {len(documents)} documents to Pinecone")
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
        """Search for similar documents in Pinecone."""
        try:
            query_embedding = self.embed_text(query)

            # Build filter
            pinecone_filter = None
            if filter_metadata:
                pinecone_filter = filter_metadata

            # Query index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=pinecone_filter,
                include_metadata=True
            )

            search_results = []
            for match in results.matches:
                metadata = match.metadata.copy()
                content = metadata.pop("content", "")

                doc = Document(
                    content=content,
                    metadata=metadata,
                    id=match.id
                )

                search_results.append(SearchResult(
                    document=doc,
                    score=match.score
                ))

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document in Pinecone."""
        try:
            embedding = document.embedding or self.embed_text(document.content)

            self.index.upsert(
                vectors=[(
                    document_id,
                    embedding,
                    {
                        **document.metadata,
                        "content": document.content
                    }
                )]
            )

            logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "name": self.collection_name,
                "document_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "backend": "Pinecone"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
