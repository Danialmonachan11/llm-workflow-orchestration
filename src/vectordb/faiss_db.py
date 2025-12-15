"""FAISS implementation for vector storage."""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import pickle
import os
from loguru import logger

from .base import VectorDBBase, Document, SearchResult


class FAISSStore(VectorDBBase):
    """FAISS implementation of vector database."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[Any] = None,
        dimension: int = 768,
        index_type: str = "flat",
        persist_directory: str = "./faiss_db"
    ):
        """
        Initialize FAISS store.

        Args:
            collection_name: Name of the collection
            embedding_model: Embedding model for generating vectors
            dimension: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            persist_directory: Directory to persist index
        """
        super().__init__(collection_name, embedding_model)

        self.dimension = dimension
        self.index_type = index_type
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize FAISS index
        self.index = self._create_index()
        self.documents: Dict[int, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.next_id = 0

        # Try to load existing index
        self._load_index()

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "flat":
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def _get_index_path(self) -> str:
        """Get path for index file."""
        return os.path.join(
            self.persist_directory,
            f"{self.collection_name}.index"
        )

    def _get_metadata_path(self) -> str:
        """Get path for metadata file."""
        return os.path.join(
            self.persist_directory,
            f"{self.collection_name}.metadata"
        )

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self._get_index_path())

            metadata = {
                "documents": self.documents,
                "id_to_index": self.id_to_index,
                "next_id": self.next_id
            }
            with open(self._get_metadata_path(), 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved FAISS index to {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)

                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                self.documents = metadata["documents"]
                self.id_to_index = metadata["id_to_index"]
                self.next_id = metadata["next_id"]

                logger.info(f"Loaded FAISS index from {self.persist_directory}")
            except Exception as e:
                logger.error(f"Error loading index: {e}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to FAISS index."""
        try:
            ids = []
            embeddings = []

            for doc in documents:
                doc_id = doc.id or f"doc_{self.next_id}"
                ids.append(doc_id)

                # Generate embedding
                embedding = doc.embedding or self.embed_text(doc.content)
                embeddings.append(embedding)

                # Store document
                self.documents[self.next_id] = doc
                self.id_to_index[doc_id] = self.next_id
                self.next_id += 1

            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)

            # Save to disk
            self._save_index()

            logger.info(f"Added {len(documents)} documents to FAISS")
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
        """Search for similar documents in FAISS."""
        try:
            query_embedding = np.array(
                [self.embed_text(query)],
                dtype=np.float32
            )

            distances, indices = self.index.search(query_embedding, top_k)

            search_results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # No result found
                    continue

                doc = self.documents.get(int(idx))
                if doc is None:
                    continue

                # Apply metadata filter if provided
                if filter_metadata:
                    if not all(
                        doc.metadata.get(k) == v
                        for k, v in filter_metadata.items()
                    ):
                        continue

                search_results.append(SearchResult(
                    document=doc,
                    score=1.0 / (1.0 + float(distances[0][i])),  # Convert distance to similarity
                    distance=float(distances[0][i])
                ))

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (marks as deleted, doesn't rebuild index)."""
        try:
            for doc_id in document_ids:
                if doc_id in self.id_to_index:
                    idx = self.id_to_index[doc_id]
                    del self.documents[idx]
                    del self.id_to_index[doc_id]

            self._save_index()
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document in FAISS."""
        try:
            if document_id in self.id_to_index:
                idx = self.id_to_index[document_id]
                self.documents[idx] = document
                self._save_index()
                logger.info(f"Updated document {document_id}")
                return True
            else:
                logger.warning(f"Document {document_id} not found")
                return False
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get FAISS collection statistics."""
        return {
            "name": self.collection_name,
            "document_count": len(self.documents),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "backend": "FAISS"
        }
