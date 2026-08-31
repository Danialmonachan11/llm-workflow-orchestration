"""Retrieval Agent with vector database integration.

Supports three retrieval-quality upgrades layered on top of the original
single-backend dense search, all optional and off unless configured (so
existing callers that only pass `vector_db` keep their exact original
behaviour):

- Dense + sparse hybrid search: an optional BM25 keyword index
  (`bm25_corpus`) is fused with dense vector results via Reciprocal Rank
  Fusion (see `..retrieval.fusion`).
- Multi-backend combination: an optional `secondary_vector_db` (e.g. a
  Qdrant store alongside a primary Chroma store) is queried too and fused
  into the same ranking -- this is what actually "combines" two vector
  stores, rather than treating them as interchangeable alternatives.
- Cross-encoder reranking: once dense/sparse/multi-backend candidates are
  fused, an optional cross-encoder model reorders the fused candidates by
  a real relevance score (query, document) pair-wise, before the final
  top_k is taken.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base_agent import BaseAgent, AgentConfig, AgentMessage, AgentResponse
from ..vectordb.base import VectorDBBase, SearchResult, Document
from ..retrieval.fusion import reciprocal_rank_fusion


class RetrievalAgent(BaseAgent):
    """Agent specialized in retrieving relevant information, optionally via
    hybrid (dense + sparse) search across one or two vector-store backends,
    with cross-encoder reranking of the fused candidates."""

    def __init__(
        self,
        config: AgentConfig,
        vector_db: VectorDBBase,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        secondary_vector_db: Optional[VectorDBBase] = None,
        bm25_corpus: Optional[List[Document]] = None,
        use_reranking: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion_candidate_multiplier: int = 4,
    ):
        """
        Initialize Retrieval agent.

        Args:
            config: Agent configuration
            vector_db: Primary vector database instance
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score for results.
                Only applied in the plain single-backend, no-fusion,
                no-reranking path -- RRF-fused and cross-encoder-reranked
                scores are not on the same [0, 1] cosine-similarity scale,
                so this threshold is not meaningful once either is active
                (top_k alone bounds the result count in that case).
            secondary_vector_db: Optional second vector store (e.g. Qdrant
                alongside a primary Chroma store) queried and fused
                alongside the primary -- a genuine combination of two
                backends, not a choice between them.
            bm25_corpus: Optional list of Documents to build a BM25
                keyword index from, fused with dense results for hybrid
                (dense + sparse) search.
            use_reranking: If True, rerank the fused candidates with a
                cross-encoder before truncating to top_k.
            rerank_model: sentence-transformers CrossEncoder model name.
            fusion_candidate_multiplier: how many candidates (top_k *
                multiplier) to pull from each retrieval source before
                fusion/reranking, so the final top_k is chosen from a
                wide enough candidate pool.
        """
        super().__init__(config)
        self.vector_db = vector_db
        self.secondary_vector_db = secondary_vector_db
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_reranking = use_reranking
        self.fusion_candidate_multiplier = fusion_candidate_multiplier

        self._bm25 = None
        self._bm25_documents: List[Document] = []
        if bm25_corpus:
            self._build_bm25_index(bm25_corpus)

        self._reranker = None
        self._rerank_model_name = rerank_model
        if use_reranking:
            self._load_reranker()

        self._hybrid_enabled = bool(secondary_vector_db or bm25_corpus)

        logger.info(
            f"Initialized Retrieval agent with {vector_db.__class__.__name__}"
            f"{' + ' + secondary_vector_db.__class__.__name__ if secondary_vector_db else ''}"
            f"{' + BM25' if self._bm25 else ''}"
            f"{' + reranking' if use_reranking else ''}"
        )

    def _build_bm25_index(self, corpus: List[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_documents = corpus
        tokenized = [doc.content.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"Built BM25 index over {len(corpus)} documents")

    def _load_reranker(self) -> None:
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(self._rerank_model_name)
        logger.info(f"Loaded cross-encoder reranker: {self._rerank_model_name}")

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(document=self._bm25_documents[i], score=float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0
        ]

    def _rerank(self, query: str, candidates: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not candidates:
            return candidates
        if self._reranker is None:
            return candidates[:top_k]

        pairs = [(query, c.document.content) for c in candidates]
        scores = self._reranker.predict(pairs)

        reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(document=result.document, score=float(score), distance=result.distance)
            for result, score in reranked[:top_k]
        ]

    def _retrieve(self, query: str, filter_metadata: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Run the full retrieval pipeline: gather candidates from every
        configured source, fuse them (if more than one source is active),
        rerank (if configured), and return the final top_k."""
        candidate_k = self.top_k * self.fusion_candidate_multiplier

        if not self._hybrid_enabled and not self.use_reranking:
            # Exact original behaviour: single dense search, threshold-filtered.
            results = self.vector_db.search(query=query, top_k=self.top_k, filter_metadata=filter_metadata)
            return [r for r in results if r.score >= self.similarity_threshold]

        result_lists: List[List[SearchResult]] = [
            self.vector_db.search(query=query, top_k=candidate_k, filter_metadata=filter_metadata)
        ]
        if self.secondary_vector_db is not None:
            result_lists.append(
                self.secondary_vector_db.search(query=query, top_k=candidate_k, filter_metadata=filter_metadata)
            )
        if self._bm25 is not None:
            result_lists.append(self._bm25_search(query, candidate_k))

        fused = reciprocal_rank_fusion(result_lists) if len(result_lists) > 1 else result_lists[0]

        if self.use_reranking:
            return self._rerank(query, fused, self.top_k)
        return fused[: self.top_k]

    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process message by retrieving relevant information.

        Args:
            message: Input message containing query

        Returns:
            Retrieved information
        """
        if not self.validate_input(message):
            return AgentResponse(
                content="",
                success=False,
                error="Invalid input message"
            )

        try:
            # Add to history
            self.add_to_history(message)

            filtered_results = self._retrieve(message.content, message.metadata.get("filter"))

            # Format results
            formatted_content = self._format_results(filtered_results)

            return AgentResponse(
                content=formatted_content,
                success=True,
                metadata={
                    "num_results": len(filtered_results),
                    "hybrid": self._hybrid_enabled,
                    "reranked": self.use_reranking,
                    "results": [
                        {
                            "content": r.document.content,
                            "score": r.score,
                            "metadata": r.document.metadata
                        }
                        for r in filtered_results
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )

    def _format_results(self, results: List[SearchResult]) -> str:
        """
        Format search results as text.

        Args:
            results: List of search results

        Returns:
            Formatted text
        """
        if not results:
            return "No relevant information found."

        formatted = ["Retrieved Information:\n"]

        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. [Score: {result.score:.2f}]")
            formatted.append(f"   {result.document.content}")

            if result.document.metadata:
                formatted.append(f"   Metadata: {result.document.metadata}")

            formatted.append("")

        return "\n".join(formatted)

    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute retrieval task.

        Args:
            task: Retrieval query
            context: Optional context with filters

        Returns:
            Retrieved results
        """
        # Create message with context as metadata
        message = AgentMessage(
            content=task,
            role="user",
            metadata=context or {}
        )

        return await self.process(message)

    async def retrieve_and_augment(
        self,
        query: str,
        augmentation_template: Optional[str] = None
    ) -> str:
        """
        Retrieve information and augment it with a template.

        Args:
            query: Search query
            augmentation_template: Template for augmentation

        Returns:
            Augmented content
        """
        # Retrieve results
        response = await self.execute_task(query)

        if not response.success:
            return ""

        # Use default template if not provided
        if not augmentation_template:
            augmentation_template = (
                "Based on the following retrieved information:\n\n"
                "{retrieved_info}\n\n"
                "Query: {query}"
            )

        # Augment with template
        augmented = augmentation_template.format(
            retrieved_info=response.content,
            query=query
        )

        return augmented

    def update_retrieval_params(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Update retrieval parameters.

        Args:
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        if top_k is not None:
            self.top_k = top_k
            logger.info(f"Updated top_k to {top_k}")

        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            logger.info(f"Updated similarity_threshold to {similarity_threshold}")
