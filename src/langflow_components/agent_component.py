"""Langflow component for LLM agents."""

from langflow import CustomComponent
from typing import Dict, Any
import asyncio

from ..agents import LLMAgent, RetrievalAgent, AgentConfig, AgentRole, AgentMessage


class AgentComponent(CustomComponent):
    """Custom Langflow component for LLM agents."""

    display_name = "LLM Agent"
    description = "Execute tasks using LLM agents"
    documentation = "https://github.com/yourusername/llm-workflow-orchestration"

    def build_config(self) -> Dict[str, Any]:
        """Build component configuration."""
        return {
            "agent_type": {
                "display_name": "Agent Type",
                "options": ["LLM Agent", "Retrieval Agent"],
                "value": "LLM Agent",
            },
            "agent_name": {
                "display_name": "Agent Name",
                "value": "assistant",
            },
            "role": {
                "display_name": "Agent Role",
                "options": ["generator", "analyzer", "summarizer", "validator"],
                "value": "generator",
            },
            "provider": {
                "display_name": "LLM Provider",
                "options": ["openai", "anthropic", "cohere"],
                "value": "openai",
            },
            "model_name": {
                "display_name": "Model Name",
                "value": "gpt-3.5-turbo",
            },
            "api_key": {
                "display_name": "API Key",
                "password": True,
                "value": "",
            },
            "system_prompt": {
                "display_name": "System Prompt",
                "multiline": True,
                "value": "You are a helpful AI assistant.",
            },
            "temperature": {
                "display_name": "Temperature",
                "value": 0.7,
            },
            "max_tokens": {
                "display_name": "Max Tokens",
                "value": 2000,
            },
            "task": {
                "display_name": "Task",
                "multiline": True,
                "value": "",
            },
        }

    def build(
        self,
        agent_type: str,
        agent_name: str,
        role: str,
        provider: str,
        model_name: str,
        api_key: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        task: str,
    ) -> Dict[str, Any]:
        """
        Execute agent task.

        Args:
            agent_type: Type of agent
            agent_name: Name of the agent
            role: Agent role
            provider: LLM provider
            model_name: Model to use
            api_key: API key
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            task: Task to execute

        Returns:
            Agent response
        """
        # Map role string to enum
        role_map = {
            "generator": AgentRole.GENERATOR,
            "analyzer": AgentRole.ANALYZER,
            "summarizer": AgentRole.SUMMARIZER,
            "validator": AgentRole.VALIDATOR,
        }
        agent_role = role_map.get(role, AgentRole.GENERATOR)

        # Create agent config
        config = AgentConfig(
            name=agent_name,
            role=agent_role,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )

        # Initialize agent
        if agent_type == "LLM Agent":
            agent = LLMAgent(
                config=config,
                api_key=api_key,
                provider=provider
            )
        else:
            # For retrieval agent, would need vector DB instance
            # Simplified for now
            return {"error": "Retrieval agent requires vector database connection"}

        # Execute task
        response = asyncio.run(agent.execute_task(task))

        return {
            "success": response.success,
            "content": response.content,
            "agent_name": agent_name,
            "error": response.error,
            "metadata": response.metadata
        }
