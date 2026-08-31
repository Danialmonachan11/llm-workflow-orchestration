"""FastAPI service layer for the LLM workflow orchestration framework.

Exposes the LangGraph-backed orchestrator (see `src/agents/orchestrator.py`)
over HTTP, with Pydantic request/response schemas -- request bodies and
response bodies are validated models, not raw dicts.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .bootstrap import build_orchestrator
from ..agents.orchestrator import WorkflowType


_orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    _orchestrator = build_orchestrator()
    yield
    _orchestrator = None


app = FastAPI(
    title="LLM Workflow Orchestration",
    description="Multi-agent orchestration (LangGraph-backed) with hybrid retrieval and cross-encoder reranking.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Request body for POST /query -- runs the RAG workflow end to end."""

    query: str = Field(..., min_length=1, description="The question to answer")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filter passed through to retrieval"
    )


class StepResult(BaseModel):
    agent: str
    action: str
    success: bool
    output: str


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    success: bool
    answer: str
    retrieved_context: str
    step_results: List[StepResult]
    error: Optional[str] = None


class WorkflowExecuteRequest(BaseModel):
    """Request body for POST /workflows/{workflow_type}/execute."""

    workflow_name: str = Field(..., description="Name of a registered workflow")
    input: str = Field(..., min_length=1, description="Initial input to the workflow")
    context: Optional[Dict[str, Any]] = None


class WorkflowExecuteResponse(BaseModel):
    success: bool
    outputs: Dict[str, Any]
    step_results: List[StepResult]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    agents: List[str]
    workflows: List[str]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return HealthResponse(
        status="ok",
        agents=_orchestrator.list_agents(),
        workflows=_orchestrator.list_workflows(),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the registered 'rag_qa' workflow end to end: hybrid retrieval
    (dense + BM25, optionally a second vector backend), cross-encoder
    reranking, then generation grounded in the retrieved context."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    initial_input = request.query
    context = {"filter": request.filter_metadata} if request.filter_metadata else {}

    result = await _orchestrator.execute_workflow(
        workflow_name="rag_qa", initial_input=initial_input, context=context
    )

    return QueryResponse(
        success=result.success,
        answer=result.outputs.get("generation", ""),
        retrieved_context=result.outputs.get("retrieval", ""),
        step_results=[StepResult(**sr) for sr in result.step_results],
        error=result.error,
    )


@app.post("/workflows/{workflow_type}/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(workflow_type: str, request: WorkflowExecuteRequest) -> WorkflowExecuteResponse:
    """Execute a registered sequential or parallel workflow by name."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    valid_types = {WorkflowType.SEQUENTIAL.value, WorkflowType.PARALLEL.value}
    if workflow_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported workflow_type '{workflow_type}'; use one of {sorted(valid_types)}",
        )

    result = await _orchestrator.execute_workflow(
        workflow_name=request.workflow_name, initial_input=request.input, context=request.context
    )

    return WorkflowExecuteResponse(
        success=result.success,
        outputs=result.outputs,
        step_results=[StepResult(**sr) for sr in result.step_results],
        error=result.error,
    )
