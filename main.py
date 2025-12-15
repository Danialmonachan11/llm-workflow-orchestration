"""Main entry point for LLM Workflow Orchestration demo."""

import asyncio
import os
from dotenv import load_dotenv

from src import (
    LLMAgent,
    AgentConfig,
    AgentRole,
    setup_logging,
    load_config
)

# Load environment variables
load_dotenv()


async def demo():
    """Run a simple demo of the LLM workflow system."""

    # Setup logging
    setup_logging(log_level="INFO")

    # Load configuration
    config = load_config()

    print("=" * 80)
    print("LLM Workflow Orchestration - Demo")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file.")
        return

    # Create a simple agent
    print("\nInitializing LLM Agent...")
    agent_config = AgentConfig(
        name="demo_assistant",
        role=AgentRole.GENERATOR,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant that explains technical concepts clearly."
    )

    agent = LLMAgent(
        config=agent_config,
        api_key=api_key,
        provider="openai"
    )

    # Interactive demo
    print("\n" + "-" * 80)
    print("Agent initialized successfully!")
    print("Ask questions or type 'quit' to exit.")
    print("-" * 80 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Process with agent
            print("\nAssistant: ", end="", flush=True)
            response = await agent.execute_task(user_input)

            if response.success:
                print(response.content)
            else:
                print(f"Error: {response.error}")

            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    try:
        asyncio.run(demo())
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
