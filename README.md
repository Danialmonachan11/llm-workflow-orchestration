# LLM Workflow Orchestration
*https://danialmonachan11.github.io/demo-llm.html*

A powerful framework for building and orchestrating Agentic LLM workflows with vector database integration for Generative AI applications.

## Features

- **Multi-Agent Orchestration**: LangGraph-backed `StateGraph` execution engine (sequential chain, parallel fan-out/fan-in, and a retrieve->generate RAG graph) behind one `AgentOrchestrator` interface
- **LangChain LLM clients**: `LLMAgent` uses LangChain's `ChatOpenAI` / `ChatAnthropic` / `ChatCohere` wrappers (OpenAI-compatible endpoints, including OpenRouter, work via `base_url`)
- **Hybrid retrieval**: dense vector search fused with BM25 keyword search via Reciprocal Rank Fusion, plus optional dual-backend combination (query Chroma and Qdrant together, not as alternatives)
- **Cross-encoder reranking**: fused candidates reordered by a `sentence-transformers` cross-encoder before the final top-k is returned
- **RAGAS evaluation**: `src/evaluation/ragas_eval.py` scores the real RAG pipeline (faithfulness, answer relevancy, context precision/recall) against a golden question set
- **FastAPI service**: `src/api/app.py` exposes `/query` and `/workflows/{type}/execute` with Pydantic request/response schemas
- **Docker packaging**: `Dockerfile` + `docker-compose.yml` (app + Qdrant) for reproducible deployment
- **Vector Database Integration**: Support for ChromaDB, FAISS, Pinecone, and Qdrant
- **Langflow Integration**: Custom components for visual workflow building (a separate, unrelated project to LangGraph — see the Langflow Integration section below)
- **Configuration Management**: YAML/JSON configuration with environment variable overrides
- **Production-Ready**: Comprehensive logging, error handling, and async support

## Architecture

```
llm-workflow-orchestration/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Base agent class
│   │   ├── llm_agent.py     # LLM-powered agent
│   │   ├── retrieval_agent.py   # RAG agent
│   │   └── orchestrator.py  # Multi-agent orchestrator
│   ├── vectordb/            # Vector database integrations
│   │   ├── base.py          # Base vector DB interface
│   │   ├── chroma_db.py     # ChromaDB implementation
│   │   ├── faiss_db.py      # FAISS implementation
│   │   ├── pinecone_db.py   # Pinecone implementation
│   │   └── qdrant_db.py     # Qdrant implementation
│   ├── langflow_components/ # Custom Langflow components
│   │   ├── vector_db_component.py
│   │   ├── agent_component.py
│   │   └── orchestrator_component.py
│   └── utils/               # Utilities
│       ├── config.py        # Configuration management
│       └── logging_config.py
├── examples/                # Usage examples
├── config/                  # Configuration files
└── tests/                   # Test suite
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-workflow-orchestration.git
cd llm-workflow-orchestration
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### Basic Agent Usage

```python
import asyncio
from src.agents import LLMAgent, AgentConfig, AgentRole

async def main():
    config = AgentConfig(
        name="assistant",
        role=AgentRole.GENERATOR,
        model_name="gpt-3.5-turbo",
        system_prompt="You are a helpful AI assistant."
    )

    agent = LLMAgent(
        config=config,
        api_key="your-api-key",
        provider="openai"
    )

    response = await agent.execute_task(
        "Explain transformers in machine learning"
    )

    print(response.content)

asyncio.run(main())
```

### Vector Database Usage

```python
from sentence_transformers import SentenceTransformer
from src.vectordb import ChromaDBStore, Document

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create vector store
vector_db = ChromaDBStore(
    collection_name="my_docs",
    embedding_model=embedding_model
)

# Add documents
documents = [
    Document(
        content="Your document content here",
        metadata={"source": "example"}
    )
]
vector_db.add_documents(documents)

# Search
results = vector_db.search("your query", top_k=5)
for result in results:
    print(f"Score: {result.score}, Content: {result.document.content}")
```

### RAG Workflow

```python
import asyncio
from src.agents import AgentOrchestrator, WorkflowConfig, WorkflowType

async def main():
    orchestrator = AgentOrchestrator()

    # Register agents (retrieval + generation)
    orchestrator.register_agent(retrieval_agent)
    orchestrator.register_agent(generation_agent)

    # Define RAG workflow
    workflow = WorkflowConfig(
        name="rag_qa",
        workflow_type=WorkflowType.RAG,
        steps=[
            WorkflowStep(agent_name="retriever", action="retrieve"),
            WorkflowStep(agent_name="generator", action="generate")
        ],
        initial_step="retriever"
    )

    orchestrator.register_workflow(workflow)

    # Execute
    result = await orchestrator.execute_workflow(
        workflow_name="rag_qa",
        initial_input="Your question here"
    )

    print(result.outputs["generation"])

asyncio.run(main())
```

## Configuration

### YAML Configuration

```yaml
# config/config.yaml
vector_db:
  type: chromadb
  collection_name: llm_workflows
  persist_directory: ./data/vector_db

llm:
  provider: openai
  model_name: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 2000

agents:
  research_agent:
    role: analyzer
    system_prompt: "You are a research assistant."

workflows:
  rag_workflow:
    type: rag
    agents:
      - retrieval_agent
      - creative_agent
```

### Environment Variables

```bash
# .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
VECTOR_DB_TYPE=chromadb
LLM_PROVIDER=openai
```

## Examples

### 1. Basic Agent
```bash
python examples/basic_agent_example.py
```

### 2. Vector Database
```bash
python examples/vector_db_example.py
```

### 3. RAG Workflow
```bash
python examples/rag_workflow_example.py
```

### 4. Multi-Agent Orchestration
```bash
python examples/multi_agent_example.py
```

## Langflow Integration

**Note on naming:** Langflow (below) and LangGraph (the orchestration
engine in `src/agents/orchestrator.py`) are two different, unrelated
projects from the LangChain ecosystem -- Langflow is a visual drag-and-drop
flow builder; LangGraph is the state-graph library that actually executes
every workflow in this repo. This section is about the former; the
orchestrator's real execution engine is documented under "Workflow Types"
below.

This project includes custom Langflow components for visual workflow building:

1. **VectorDBComponent**: Store and retrieve documents
2. **AgentComponent**: Execute LLM agent tasks
3. **OrchestratorComponent**: Orchestrate multi-agent workflows

To use in Langflow:
1. Copy components from `src/langflow_components/` to your Langflow custom components directory
2. Restart Langflow
3. Find components in the custom components section

## Workflow Types

### Sequential
Agents execute one after another, with each agent's output feeding into the next.

```python
WorkflowType.SEQUENTIAL
# Agent1 -> Agent2 -> Agent3
```

### Parallel
Multiple agents execute simultaneously on the same input.

```python
WorkflowType.PARALLEL
# Input -> [Agent1, Agent2, Agent3] -> Combined Output
```

### RAG (Retrieval-Augmented Generation)
Combines retrieval from vector database with LLM generation.

```python
WorkflowType.RAG
# Input -> Retrieval Agent -> Generation Agent (with context) -> Output
```

## Supported Vector Databases

- **ChromaDB**: Easy-to-use, local-first database
- **FAISS**: Facebook's similarity search library
- **Pinecone**: Managed vector database service
- **Qdrant**: Open-source vector search engine

## Supported LLM Providers

- **OpenAI**: GPT-3.5, GPT-4 models
- **Anthropic**: Claude models
- **Cohere**: Command models

## API Reference

### Agent Classes

#### BaseAgent
Base class for all agents.

Methods:
- `process(message)`: Process a message
- `execute_task(task, context)`: Execute a task
- `clear_history()`: Clear conversation history

#### LLMAgent
Agent powered by LLMs.

```python
LLMAgent(
    config: AgentConfig,
    api_key: str,
    provider: str  # 'openai', 'anthropic', 'cohere'
)
```

#### RetrievalAgent
Agent for retrieval-augmented generation.

```python
RetrievalAgent(
    config: AgentConfig,
    vector_db: VectorDBBase,
    top_k: int = 5,
    similarity_threshold: float = 0.7
)
```

### Vector Database Classes

All vector DB classes implement:
- `add_documents(documents)`: Add documents
- `search(query, top_k, filter_metadata)`: Search for similar documents
- `delete_documents(document_ids)`: Delete documents
- `update_document(document_id, document)`: Update a document
- `get_collection_stats()`: Get statistics

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
ruff check src/
```

## Project Structure Details

- **agents/**: Multi-agent system with orchestration
- **vectordb/**: Pluggable vector database backends
- **langflow_components/**: Visual workflow components
- **utils/**: Configuration and logging utilities
- **examples/**: Comprehensive usage examples
- **config/**: Configuration templates

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with LangChain, LangGraph, RAGAS, FastAPI, Transformers, and modern LLM APIs
- Inspired by agent-based architectures and RAG patterns
- Vector database integrations for efficient retrieval

## Contact

For questions or support, please open an issue on GitHub.

## Roadmap

- [ ] Add more vector database backends (Weaviate, Milvus)
- [ ] Implement conditional workflows
- [ ] Add streaming support for LLM responses
- [ ] Web UI for workflow management
- [ ] More LLM provider integrations
- [ ] Enhanced monitoring and observability
- [ ] Agent memory and state persistence
- [ ] Tool use and function calling support

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_workflow_orchestration,
  title={LLM Workflow Orchestration},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-workflow-orchestration}
}
```
