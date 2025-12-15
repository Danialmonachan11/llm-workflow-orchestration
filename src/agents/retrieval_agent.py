"""Retrieval Agent with vector database integration."""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base_agent import BaseAgent, AgentConfig, AgentMessage, AgentResponse
from ..vectordb.base import VectorDBBase, SearchResult


class RetrievalAgent(BaseAgent):
    """Agent specialized in retrieving relevant information from vector database."""

    def __init__(
        self,
        config: AgentConfig,
        vector_db: VectorDBBase,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize Retrieval agent.

        Args:
            config: Agent configuration
            vector_db: Vector database instance
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score for results
        """
        super().__init__(config)
        self.vector_db = vector_db
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        logger.info(f"Initialized Retrieval agent with {vector_db.__class__.__name__}")

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

            # Search vector database
            results = self.vector_db.search(
                query=message.content,
                top_k=self.top_k,
                filter_metadata=message.metadata.get("filter")
            )

            # Filter by similarity threshold
            filtered_results = [
                r for r in results
                if r.score >= self.similarity_threshold
            ]

            # Format results
            formatted_content = self._format_results(filtered_results)

            return AgentResponse(
                content=formatted_content,
                success=True,
                metadata={
                    "num_results": len(filtered_results),
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
