"""Base agent class for LLM workflow orchestration."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class AgentRole(Enum):
    """Enumeration of agent roles."""
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    RETRIEVER = "retriever"
    SUMMARIZER = "summarizer"
    VALIDATOR = "validator"
    CUSTOM = "custom"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    role: AgentRole
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message format for agent communication."""

    content: str
    role: str = "user"  # user, assistant, system
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_name: Optional[str] = None


@dataclass
class AgentResponse:
    """Response from an agent."""

    content: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig):
        """
        Initialize agent with configuration.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.name = config.name
        self.role = config.role
        self.conversation_history: List[AgentMessage] = []

        logger.info(f"Initialized agent: {self.name} with role {self.role.value}")

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process a message and generate a response.

        Args:
            message: Input message

        Returns:
            Agent response
        """
        pass

    def add_to_history(self, message: AgentMessage):
        """
        Add a message to conversation history.

        Args:
            message: Message to add
        """
        self.conversation_history.append(message)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info(f"Cleared history for agent: {self.name}")

    def get_history(self) -> List[AgentMessage]:
        """
        Get conversation history.

        Returns:
            List of messages
        """
        return self.conversation_history

    def update_config(self, **kwargs):
        """
        Update agent configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} for agent {self.name}")

    @abstractmethod
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute a specific task.

        Args:
            task: Task description
            context: Optional context information

        Returns:
            Task execution result
        """
        pass

    def validate_input(self, message: AgentMessage) -> bool:
        """
        Validate input message.

        Args:
            message: Message to validate

        Returns:
            True if valid, False otherwise
        """
        if not message.content or not isinstance(message.content, str):
            logger.warning(f"Invalid message content for agent {self.name}")
            return False
        return True

    def format_prompt(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format prompt with system prompt and context.

        Args:
            message: User message
            context: Optional context

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        if self.config.system_prompt:
            prompt_parts.append(f"System: {self.config.system_prompt}")

        if context:
            prompt_parts.append(f"Context: {context}")

        prompt_parts.append(f"User: {message}")

        return "\n\n".join(prompt_parts)

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"<{self.__class__.__name__}(name='{self.name}', role='{self.role.value}')>"
