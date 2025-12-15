"""Example of multi-agent orchestration with sequential and parallel workflows."""

import asyncio
import os
from dotenv import load_dotenv

from src.agents import (
    LLMAgent,
    AgentOrchestrator,
    AgentConfig,
    AgentRole,
    WorkflowConfig,
    WorkflowStep,
    WorkflowType
)
from src.utils import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(log_level="INFO")


async def sequential_workflow_example():
    """Example of sequential workflow: Research -> Analyze -> Summarize."""

    print("\n" + "=" * 80)
    print("SEQUENTIAL WORKFLOW EXAMPLE")
    print("=" * 80)

    # Create agents
    researcher = LLMAgent(
        config=AgentConfig(
            name="researcher",
            role=AgentRole.ANALYZER,
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="You are a research assistant. Gather and present factual information on topics."
        ),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    analyzer = LLMAgent(
        config=AgentConfig(
            name="analyzer",
            role=AgentRole.ANALYZER,
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            system_prompt="You analyze information and identify key insights and patterns."
        ),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    summarizer = LLMAgent(
        config=AgentConfig(
            name="summarizer",
            role=AgentRole.SUMMARIZER,
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            system_prompt="You create concise, clear summaries that capture essential information."
        ),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    # Create orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(researcher)
    orchestrator.register_agent(analyzer)
    orchestrator.register_agent(summarizer)

    # Define sequential workflow
    workflow = WorkflowConfig(
        name="research_workflow",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(agent_name="researcher", action="research"),
            WorkflowStep(agent_name="analyzer", action="analyze"),
            WorkflowStep(agent_name="summarizer", action="summarize")
        ],
        initial_step="researcher"
    )

    orchestrator.register_workflow(workflow)

    # Execute workflow
    query = "What are the latest trends in large language models in 2024?"

    print(f"\nQuery: {query}\n")
    print("Processing through: Researcher -> Analyzer -> Summarizer\n")

    result = await orchestrator.execute_workflow(
        workflow_name="research_workflow",
        initial_input=query
    )

    if result.success:
        print("\nWorkflow Results:")
        print("-" * 80)
        for i, step in enumerate(result.step_results, 1):
            print(f"\nStep {i}: {step['agent']}")
            print(f"Output: {step['output'][:200]}...")
            print("-" * 80)

        print("\n\nFinal Output (Summarizer):")
        print("=" * 80)
        print(result.outputs.get("summarizer", ""))
        print("=" * 80)
    else:
        print(f"Error: {result.error}")


async def parallel_workflow_example():
    """Example of parallel workflow: Multiple perspectives on same topic."""

    print("\n\n" + "=" * 80)
    print("PARALLEL WORKFLOW EXAMPLE")
    print("=" * 80)

    # Create agents with different perspectives
    technical_expert = LLMAgent(
        config=AgentConfig(
            name="technical_expert",
            role=AgentRole.ANALYZER,
            model_name="gpt-3.5-turbo",
            temperature=0.4,
            system_prompt="You are a technical expert. Provide detailed technical analysis."
        ),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    business_analyst = LLMAgent(
        config=AgentConfig(
            name="business_analyst",
            role=AgentRole.ANALYZER,
            model_name="gpt-3.5-turbo",
            temperature=0.6,
            system_prompt="You are a business analyst. Focus on business impact and opportunities."
        ),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    creative_thinker = LLMAgent(
        config=AgentConfig(
            name="creative_thinker",
            role=AgentRole.GENERATOR,
            model_name="gpt-3.5-turbo",
            temperature=0.8,
            system_prompt="You are a creative thinker. Generate innovative ideas and applications."
        ),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider="openai"
    )

    # Create orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(technical_expert)
    orchestrator.register_agent(business_analyst)
    orchestrator.register_agent(creative_thinker)

    # Define parallel workflow
    workflow = WorkflowConfig(
        name="multi_perspective",
        workflow_type=WorkflowType.PARALLEL,
        steps=[
            WorkflowStep(agent_name="technical_expert", action="analyze"),
            WorkflowStep(agent_name="business_analyst", action="analyze"),
            WorkflowStep(agent_name="creative_thinker", action="brainstorm")
        ],
        initial_step="technical_expert"
    )

    orchestrator.register_workflow(workflow)

    # Execute workflow
    query = "Analyze the impact of generative AI on software development"

    print(f"\nQuery: {query}\n")
    print("Getting perspectives from: Technical Expert | Business Analyst | Creative Thinker\n")

    result = await orchestrator.execute_workflow(
        workflow_name="multi_perspective",
        initial_input=query
    )

    if result.success:
        print("\nWorkflow Results:")
        print("=" * 80)

        agents_order = ["technical_expert", "business_analyst", "creative_thinker"]
        titles = ["Technical Perspective", "Business Perspective", "Creative Perspective"]

        for agent_name, title in zip(agents_order, titles):
            print(f"\n{title}:")
            print("-" * 80)
            print(result.outputs.get(agent_name, "No output"))
            print()
    else:
        print(f"Error: {result.error}")


async def main():
    """Run all examples."""

    # Run sequential workflow
    await sequential_workflow_example()

    # Run parallel workflow
    await parallel_workflow_example()


if __name__ == "__main__":
    asyncio.run(main())
