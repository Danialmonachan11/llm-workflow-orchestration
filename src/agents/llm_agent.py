"""LLM Agent implementation using various LLM providers via LangChain."""

from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from loguru import logger

from .base_agent import BaseAgent, AgentConfig, AgentMessage, AgentResponse


class LLMAgent(BaseAgent):
    """Agent that uses LangChain chat models for processing."""

    def __init__(
        self,
        config: AgentConfig,
        api_key: str = "",
        provider: str = "openai",
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM agent.

        Args:
            config: Agent configuration
            api_key: API key for LLM provider
            provider: LLM provider ('openai', 'anthropic', 'cohere')
            base_url: Optional custom base URL (e.g. OpenRouter's
                'https://openrouter.ai/api/v1' with an 'openai' provider,
                so any OpenRouter-hosted model works through the same
                OpenAI-compatible LangChain client)
        """
        super().__init__(config)
        self.api_key = api_key
        self.provider = provider
        self.base_url = base_url

        # Initialize LangChain chat model based on provider
        if provider == "openai":
            self.client = ChatOpenAI(
                model=config.model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif provider == "anthropic":
            self.client = ChatAnthropic(
                model=config.model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif provider == "cohere":
            self.client = ChatCohere(
                model=config.model_name,
                cohere_api_key=api_key,
                temperature=config.temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"Initialized LLM agent with provider: {provider} (LangChain)")

    def _build_messages(self) -> List[BaseMessage]:
        """Build the LangChain message list from system prompt + conversation history."""
        messages: List[BaseMessage] = []

        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))

        for msg in self.conversation_history:
            if msg.role == "system":
                continue
            if msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            else:
                messages.append(HumanMessage(content=msg.content))

        return messages

    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process message using the configured LangChain chat model.

        Args:
            message: Input message

        Returns:
            LLM response
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

            langchain_messages = self._build_messages()
            result = await self.client.ainvoke(langchain_messages)
            response_text = result.content

            # Create response message
            response_message = AgentMessage(
                content=response_text,
                role="assistant",
                agent_name=self.name
            )
            self.add_to_history(response_message)

            return AgentResponse(
                content=response_text,
                success=True,
                metadata={"provider": self.provider}
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )

    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute a specific task.

        Args:
            task: Task description
            context: Optional context information

        Returns:
            Task execution result
        """
        # Format prompt with context
        prompt = self.format_prompt(task, context)

        # Create message
        message = AgentMessage(
            content=prompt,
            role="user"
        )

        return await self.process(message)

    def set_system_prompt(self, prompt: str):
        """
        Set system prompt for the agent.

        Args:
            prompt: System prompt
        """
        self.config.system_prompt = prompt
        logger.info(f"Updated system prompt for agent {self.name}")

    def get_token_count(self) -> int:
        """
        Estimate token count for conversation history.

        Returns:
            Approximate token count
        """
        # Rough estimation: 1 token approx 4 characters
        total_chars = sum(len(msg.content) for msg in self.conversation_history)
        return total_chars // 4
