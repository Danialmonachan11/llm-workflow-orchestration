"""LLM Agent implementation using various LLM providers."""

from typing import Dict, Any, Optional, List
from openai import OpenAI
from anthropic import Anthropic
import cohere
from loguru import logger

from .base_agent import BaseAgent, AgentConfig, AgentMessage, AgentResponse


class LLMAgent(BaseAgent):
    """Agent that uses LLMs for processing."""

    def __init__(
        self,
        config: AgentConfig,
        api_key: str = "",
        provider: str = "openai"
    ):
        """
        Initialize LLM agent.

        Args:
            config: Agent configuration
            api_key: API key for LLM provider
            provider: LLM provider ('openai', 'anthropic', 'cohere')
        """
        super().__init__(config)
        self.api_key = api_key
        self.provider = provider

        # Initialize client based on provider
        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
        elif provider == "cohere":
            self.client = cohere.Client(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"Initialized LLM agent with provider: {provider}")

    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process message using LLM.

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

            # Generate response based on provider
            if self.provider == "openai":
                response_text = await self._call_openai(message)
            elif self.provider == "anthropic":
                response_text = await self._call_anthropic(message)
            elif self.provider == "cohere":
                response_text = await self._call_cohere(message)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

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

    async def _call_openai(self, message: AgentMessage) -> str:
        """Call OpenAI API."""
        messages = []

        # Add system prompt
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        # Add conversation history
        for msg in self.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.choices[0].message.content

    async def _call_anthropic(self, message: AgentMessage) -> str:
        """Call Anthropic API."""
        messages = []

        # Add conversation history
        for msg in self.conversation_history:
            if msg.role != "system":
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        response = self.client.messages.create(
            model=self.config.model_name,
            system=self.config.system_prompt,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.content[0].text

    async def _call_cohere(self, message: AgentMessage) -> str:
        """Call Cohere API."""
        # Build conversation
        chat_history = []
        for msg in self.conversation_history[:-1]:  # Exclude current message
            chat_history.append({
                "role": "USER" if msg.role == "user" else "CHATBOT",
                "message": msg.content
            })

        response = self.client.chat(
            message=message.content,
            chat_history=chat_history,
            preamble=self.config.system_prompt,
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.text

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
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = sum(len(msg.content) for msg in self.conversation_history)
        return total_chars // 4
