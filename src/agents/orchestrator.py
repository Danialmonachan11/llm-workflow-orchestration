"""Agent orchestrator for managing complex workflows -- backed by LangGraph.

Execution engine rebuilt on `langgraph.graph.StateGraph` (2026-08-31): each
call to `execute_workflow` compiles a graph matching the workflow's type
(sequential chain, parallel fan-out/fan-in, or a two-node retrieve->generate
RAG graph) and runs it via `graph.ainvoke()`. The public API is unchanged
from the pre-LangGraph version, so every existing caller (examples, the
Langflow orchestrator component, any already-registered WorkflowConfig)
keeps working without modification -- only what happens inside
`execute_workflow` changed.
"""

from typing import Dict, Any, Optional, List, Callable, Annotated
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict
from loguru import logger

from langgraph.graph import StateGraph, START, END

from .base_agent import BaseAgent, AgentResponse


class WorkflowType(Enum):
    """Types of workflow execution patterns."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    RAG = "rag"  # Retrieval-Augmented Generation


@dataclass
class WorkflowStep:
    """Definition of a workflow step."""

    agent_name: str
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable] = None
    next_steps: List[str] = field(default_factory=list)


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    name: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep]
    initial_step: str
    max_iterations: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result from workflow execution."""

    success: bool
    outputs: Dict[str, Any]
    step_results: List[Dict[str, Any]]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- LangGraph state reducers -------------------------------------------
# Concurrent branches (the PARALLEL graph shape) can write to the same
# state key in the same superstep; LangGraph requires an explicit reducer
# for any key more than one branch might touch, otherwise it raises an
# InvalidUpdateError even when both branches would write the same value.

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


def _concat_lists(a: list, b: list) -> list:
    return a + b


def _or_bools(a: bool, b: bool) -> bool:
    return a or b


def _first_error(a: Optional[str], b: Optional[str]) -> Optional[str]:
    return a if a is not None else b


class _GraphState(TypedDict):
    """LangGraph state threaded through every workflow execution."""
    initial_input: str
    current_input: str
    context: Dict[str, Any]
    outputs: Annotated[Dict[str, Any], _merge_dicts]
    step_results: Annotated[List[Dict[str, Any]], _concat_lists]
    failed: Annotated[bool, _or_bools]
    error: Annotated[Optional[str], _first_error]


class AgentOrchestrator:
    """Orchestrator for managing multi-agent workflows (LangGraph-backed)."""

    def __init__(self):
        """Initialize orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.execution_history: List[Dict[str, Any]] = []

        logger.info("Initialized Agent Orchestrator (LangGraph-backed)")

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def register_workflow(self, workflow: WorkflowConfig):
        """Register a workflow configuration."""
        self.workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute a registered workflow by compiling and running its LangGraph graph."""
        if workflow_name not in self.workflows:
            return WorkflowResult(
                success=False, outputs={}, step_results=[],
                error=f"Workflow {workflow_name} not found",
            )

        workflow = self.workflows[workflow_name]
        logger.info(f"Executing workflow: {workflow_name} ({workflow.workflow_type.value})")

        try:
            if workflow.workflow_type == WorkflowType.SEQUENTIAL:
                graph = self._build_sequential_graph(workflow)
            elif workflow.workflow_type == WorkflowType.PARALLEL:
                graph = self._build_parallel_graph(workflow)
            elif workflow.workflow_type == WorkflowType.RAG:
                graph = self._build_rag_graph(workflow)
            else:
                return WorkflowResult(
                    success=False, outputs={}, step_results=[],
                    error=f"Workflow type {workflow.workflow_type} not implemented",
                )
        except ValueError as e:
            return WorkflowResult(success=False, outputs={}, step_results=[], error=str(e))

        initial_state: _GraphState = {
            "initial_input": initial_input,
            "current_input": initial_input,
            "context": context or {},
            "outputs": {},
            "step_results": [],
            "failed": False,
            "error": None,
        }

        try:
            final_state = await graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return WorkflowResult(success=False, outputs={}, step_results=[], error=str(e))

        return WorkflowResult(
            success=not final_state["failed"],
            outputs=final_state["outputs"],
            step_results=final_state["step_results"],
            error=final_state.get("error"),
        )

    def _get_agent_or_raise(self, agent_name: str) -> BaseAgent:
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent {agent_name} not found")
        return agent

    def _make_step_node(self, step: WorkflowStep, chain_input: bool) -> Callable:
        """Build a LangGraph node function for one workflow step.

        chain_input=True: reads/advances `current_input` (sequential
        chaining -- each step's output becomes the next step's input).
        chain_input=False: always reads `initial_input` (parallel -- every
        step runs independently against the same starting input).
        """
        agent = self._get_agent_or_raise(step.agent_name)

        async def node(state: _GraphState) -> dict:
            if state["failed"]:
                return {}

            task_input = state["current_input"] if chain_input else state["initial_input"]
            response: AgentResponse = await agent.execute_task(task_input, state["context"])

            step_result = {
                "agent": step.agent_name, "action": step.action,
                "success": response.success, "output": response.content,
            }

            if not response.success:
                return {
                    "failed": True, "error": response.error,
                    "step_results": [step_result],
                }

            update: Dict[str, Any] = {
                "outputs": {step.agent_name: response.content},
                "step_results": [step_result],
            }
            if chain_input:
                update["current_input"] = response.content
            return update

        return node

    def _build_sequential_graph(self, workflow: WorkflowConfig):
        graph = StateGraph(_GraphState)
        node_names = []

        for i, step in enumerate(workflow.steps):
            name = f"{i}_{step.agent_name}"
            graph.add_node(name, self._make_step_node(step, chain_input=True))
            node_names.append(name)

        if not node_names:
            raise ValueError("SEQUENTIAL workflow requires at least 1 step")

        graph.add_edge(START, node_names[0])
        for a, b in zip(node_names, node_names[1:]):
            graph.add_edge(a, b)
        graph.add_edge(node_names[-1], END)

        return graph.compile()

    def _build_parallel_graph(self, workflow: WorkflowConfig):
        graph = StateGraph(_GraphState)
        node_names = []

        for i, step in enumerate(workflow.steps):
            name = f"{i}_{step.agent_name}"
            graph.add_node(name, self._make_step_node(step, chain_input=False))
            node_names.append(name)

        if not node_names:
            raise ValueError("PARALLEL workflow requires at least 1 step")

        for name in node_names:
            graph.add_edge(START, name)
            graph.add_edge(name, END)

        return graph.compile()

    def _build_rag_graph(self, workflow: WorkflowConfig):
        if len(workflow.steps) < 2:
            raise ValueError("RAG workflow requires at least 2 steps (retrieval + generation)")

        retrieval_step, generation_step = workflow.steps[0], workflow.steps[1]
        retrieval_agent = self._get_agent_or_raise(retrieval_step.agent_name)
        generation_agent = self._get_agent_or_raise(generation_step.agent_name)

        async def retrieve_node(state: _GraphState) -> dict:
            response = await retrieval_agent.execute_task(state["initial_input"], state["context"])
            step_result = {
                "agent": retrieval_step.agent_name, "action": "retrieve",
                "success": response.success, "output": response.content,
            }
            if not response.success:
                return {"failed": True, "error": response.error, "step_results": [step_result]}
            return {
                "outputs": {"retrieval": response.content},
                "step_results": [step_result],
            }

        async def generate_node(state: _GraphState) -> dict:
            if state["failed"]:
                return {}
            augmented_context = {**state["context"], "retrieved_info": state["outputs"].get("retrieval", "")}
            response = await generation_agent.execute_task(state["initial_input"], augmented_context)
            step_result = {
                "agent": generation_step.agent_name, "action": "generate",
                "success": response.success, "output": response.content,
            }
            update: Dict[str, Any] = {"step_results": [step_result]}
            if not response.success:
                update["failed"] = True
                update["error"] = response.error
            else:
                update["outputs"] = {"generation": response.content}
            return update

        graph = StateGraph(_GraphState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get registered agent by name."""
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self.agents.keys())

    def list_workflows(self) -> List[str]:
        """List all registered workflows."""
        return list(self.workflows.keys())
