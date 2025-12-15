# Quick Start Guide

Get up and running with LLM Workflow Orchestration in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-workflow-orchestration.git
cd llm-workflow-orchestration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-...
```

## Run Your First Agent

```python
# save as test_agent.py
import asyncio
import os
from dotenv import load_dotenv
from src.agents import LLMAgent, AgentConfig, AgentRole

load_dotenv()

async def main():
    agent = LLMAgent(
        config=AgentConfig(
            name="assistant",
            role=AgentRole.GENERATOR,
            system_prompt="You are a helpful assistant."
        ),
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai"
    )

    response = await agent.execute_task(
        "Explain what an LLM agent is in 2 sentences"
    )

    print(response.content)

asyncio.run(main())
```

```bash
python test_agent.py
```

## Run the Interactive Demo

```bash
python main.py
```

## Try Example Workflows

```bash
# Basic agent
python examples/basic_agent_example.py

# Vector database
python examples/vector_db_example.py

# RAG workflow
python examples/rag_workflow_example.py

# Multi-agent orchestration
python examples/multi_agent_example.py
```

## Common Use Cases

### 1. Question Answering with RAG

```python
# Initialize vector DB with your documents
# Create retrieval agent + generation agent
# Execute RAG workflow
# Get answers based on your knowledge base
```

### 2. Multi-Perspective Analysis

```python
# Create multiple agents with different roles
# Execute parallel workflow
# Get diverse perspectives on same topic
```

### 3. Sequential Processing Pipeline

```python
# Create agents for: research -> analyze -> summarize
# Execute sequential workflow
# Get refined output through multiple stages
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore the [examples/](examples/) directory
3. Check [config/config.example.yaml](config/config.example.yaml) for configuration options
4. Try integrating with Langflow for visual workflow building

## Getting Help

- Check the [README.md](README.md) for API reference
- Look at examples for common patterns
- Open an issue on GitHub for bugs or questions

## Tips

- Start with basic examples before building complex workflows
- Use appropriate temperature settings (low for factual, high for creative)
- Monitor token usage to optimize costs
- Use vector databases for knowledge-intensive applications
- Combine agents with different strengths for best results
