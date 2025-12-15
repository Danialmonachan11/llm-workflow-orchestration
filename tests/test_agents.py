"""Test suite for agent functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.agents import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentMessage,
    LLMAgent
)


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_agent_config_creation(self):
        """Test creating an agent configuration."""
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.GENERATOR,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            system_prompt="Test prompt"
        )

        assert config.name == "test_agent"
        assert config.role == AgentRole.GENERATOR
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000


class TestAgentMessage:
    """Test AgentMessage dataclass."""

    def test_message_creation(self):
        """Test creating an agent message."""
        message = AgentMessage(
            content="Hello, world!",
            role="user",
            metadata={"key": "value"}
        )

        assert message.content == "Hello, world!"
        assert message.role == "user"
        assert message.metadata["key"] == "value"


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    async def process(self, message: AgentMessage):
        """Mock process method."""
        return Mock(content="Mock response", success=True)

    async def execute_task(self, task: str, context=None):
        """Mock execute_task method."""
        return Mock(content="Mock task result", success=True)


class TestBaseAgent:
    """Test BaseAgent functionality."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.GENERATOR
        )

        agent = MockAgent(config)

        assert agent.name == "test_agent"
        assert agent.role == AgentRole.GENERATOR
        assert len(agent.conversation_history) == 0

    def test_add_to_history(self):
        """Test adding messages to history."""
        config = AgentConfig(name="test", role=AgentRole.GENERATOR)
        agent = MockAgent(config)

        message = AgentMessage(content="Test", role="user")
        agent.add_to_history(message)

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0].content == "Test"

    def test_clear_history(self):
        """Test clearing conversation history."""
        config = AgentConfig(name="test", role=AgentRole.GENERATOR)
        agent = MockAgent(config)

        agent.add_to_history(AgentMessage(content="Test", role="user"))
        agent.clear_history()

        assert len(agent.conversation_history) == 0

    def test_validate_input(self):
        """Test input validation."""
        config = AgentConfig(name="test", role=AgentRole.GENERATOR)
        agent = MockAgent(config)

        valid_message = AgentMessage(content="Test", role="user")
        invalid_message = AgentMessage(content="", role="user")

        assert agent.validate_input(valid_message) is True
        assert agent.validate_input(invalid_message) is False


@pytest.mark.asyncio
class TestLLMAgent:
    """Test LLM Agent functionality."""

    @patch('openai.ChatCompletion.create')
    async def test_llm_agent_process(self, mock_openai):
        """Test LLM agent processing a message."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_openai.return_value = mock_response

        config = AgentConfig(
            name="test_llm",
            role=AgentRole.GENERATOR,
            system_prompt="Test prompt"
        )

        agent = LLMAgent(
            config=config,
            api_key="test-key",
            provider="openai"
        )

        message = AgentMessage(content="Hello", role="user")
        response = await agent.process(message)

        assert response.success is True
        assert "Test response" in response.content

    async def test_llm_agent_invalid_input(self):
        """Test LLM agent with invalid input."""
        config = AgentConfig(name="test", role=AgentRole.GENERATOR)
        agent = LLMAgent(config=config, api_key="test", provider="openai")

        invalid_message = AgentMessage(content="", role="user")
        response = await agent.process(invalid_message)

        assert response.success is False
        assert response.error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
