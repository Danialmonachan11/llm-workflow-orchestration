"""Basic example of using an LLM agent."""

import asyncio
import os
from dotenv import load_dotenv

from src.agents import LLMAgent, AgentConfig, AgentRole, AgentMessage
from src.utils import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(log_level="INFO")


async def main():
    """Run basic agent example."""

    # Create agent configuration
    config = AgentConfig(
        name="assistant",
        role=AgentRole.GENERATOR,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant specialized in explaining complex topics in simple terms."
    )

    # Initialize agent
    agent = LLMAgent(
        config=config,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    # Create message
    message = AgentMessage(
        content="Explain how transformers work in machine learning in simple terms.",
        role="user"
    )

    # Process message
    print("Sending message to agent...")
    response = await agent.process(message)

    if response.success:
        print("\nAgent Response:")
        print("-" * 80)
        print(response.content)
        print("-" * 80)
    else:
        print(f"\nError: {response.error}")

    # Execute another task
    print("\n\nExecuting another task...")
    response2 = await agent.execute_task(
        "What are the key components of a transformer architecture?"
    )

    if response2.success:
        print("\nAgent Response:")
        print("-" * 80)
        print(response2.content)
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
