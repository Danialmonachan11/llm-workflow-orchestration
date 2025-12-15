"""Agent orchestrator for managing complex workflows."""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import asyncio

from .base_agent import BaseAgent, AgentMessage, AgentResponse


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


class AgentOrchestrator:
    """Orchestrator for managing multi-agent workflows."""

    def __init__(self):
        """Initialize orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.execution_history: List[Dict[str, Any]] = []

        logger.info("Initialized Agent Orchestrator")

    def register_agent(self, agent: BaseAgent):
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def register_workflow(self, workflow: WorkflowConfig):
        """
        Register a workflow configuration.

        Args:
            workflow: Workflow configuration
        """
        self.workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a registered workflow.

        Args:
            workflow_name: Name of workflow to execute
            initial_input: Initial input to the workflow
            context: Optional context

        Returns:
            Workflow execution result
        """
        if workflow_name not in self.workflows:
            return WorkflowResult(
                success=False,
                outputs={},
                step_results=[],
                error=f"Workflow {workflow_name} not found"
            )

        workflow = self.workflows[workflow_name]
        logger.info(f"Executing workflow: {workflow_name}")

        try:
            if workflow.workflow_type == WorkflowType.SEQUENTIAL:
                return await self._execute_sequential(workflow, initial_input, context)
            elif workflow.workflow_type == WorkflowType.PARALLEL:
                return await self._execute_parallel(workflow, initial_input, context)
            elif workflow.workflow_type == WorkflowType.RAG:
                return await self._execute_rag(workflow, initial_input, context)
            else:
                return WorkflowResult(
                    success=False,
                    outputs={},
                    step_results=[],
                    error=f"Workflow type {workflow.workflow_type} not implemented"
                )

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return WorkflowResult(
                success=False,
                outputs={},
                step_results=[],
                error=str(e)
            )

    async def _execute_sequential(
        self,
        workflow: WorkflowConfig,
        initial_input: str,
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """Execute workflow sequentially."""
        step_results = []
        current_input = initial_input
        outputs = {}

        for step in workflow.steps:
            if step.agent_name not in self.agents:
                return WorkflowResult(
                    success=False,
                    outputs=outputs,
                    step_results=step_results,
                    error=f"Agent {step.agent_name} not found"
                )

            agent = self.agents[step.agent_name]
            logger.info(f"Executing step with agent: {step.agent_name}")

            # Execute agent
            response = await agent.execute_task(current_input, context)

            # Record step result
            step_result = {
                "agent": step.agent_name,
                "action": step.action,
                "success": response.success,
                "output": response.content
            }
            step_results.append(step_result)

            if not response.success:
                return WorkflowResult(
                    success=False,
                    outputs=outputs,
                    step_results=step_results,
                    error=response.error
                )

            # Update for next step
            current_input = response.content
            outputs[step.agent_name] = response.content

        return WorkflowResult(
            success=True,
            outputs=outputs,
            step_results=step_results
        )

    async def _execute_parallel(
        self,
        workflow: WorkflowConfig,
        initial_input: str,
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """Execute workflow steps in parallel."""
        tasks = []
        step_results = []
        outputs = {}

        for step in workflow.steps:
            if step.agent_name not in self.agents:
                return WorkflowResult(
                    success=False,
                    outputs=outputs,
                    step_results=step_results,
                    error=f"Agent {step.agent_name} not found"
                )

            agent = self.agents[step.agent_name]
            tasks.append(agent.execute_task(initial_input, context))

        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, (step, response) in enumerate(zip(workflow.steps, responses)):
            if isinstance(response, Exception):
                return WorkflowResult(
                    success=False,
                    outputs=outputs,
                    step_results=step_results,
                    error=str(response)
                )

            step_result = {
                "agent": step.agent_name,
                "action": step.action,
                "success": response.success,
                "output": response.content
            }
            step_results.append(step_result)

            if response.success:
                outputs[step.agent_name] = response.content

        return WorkflowResult(
            success=True,
            outputs=outputs,
            step_results=step_results
        )

    async def _execute_rag(
        self,
        workflow: WorkflowConfig,
        initial_input: str,
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """Execute Retrieval-Augmented Generation workflow."""
        step_results = []
        outputs = {}

        # Step 1: Retrieval (should be first step)
        if len(workflow.steps) < 2:
            return WorkflowResult(
                success=False,
                outputs=outputs,
                step_results=step_results,
                error="RAG workflow requires at least 2 steps (retrieval + generation)"
            )

        retrieval_step = workflow.steps[0]
        generation_step = workflow.steps[1]

        # Execute retrieval
        retrieval_agent = self.agents.get(retrieval_step.agent_name)
        if not retrieval_agent:
            return WorkflowResult(
                success=False,
                outputs=outputs,
                step_results=step_results,
                error=f"Retrieval agent {retrieval_step.agent_name} not found"
            )

        logger.info("Executing retrieval step")
        retrieval_response = await retrieval_agent.execute_task(initial_input, context)

        step_results.append({
            "agent": retrieval_step.agent_name,
            "action": "retrieve",
            "success": retrieval_response.success,
            "output": retrieval_response.content
        })

        if not retrieval_response.success:
            return WorkflowResult(
                success=False,
                outputs=outputs,
                step_results=step_results,
                error=retrieval_response.error
            )

        outputs["retrieval"] = retrieval_response.content

        # Step 2: Generation with retrieved context
        generation_agent = self.agents.get(generation_step.agent_name)
        if not generation_agent:
            return WorkflowResult(
                success=False,
                outputs=outputs,
                step_results=step_results,
                error=f"Generation agent {generation_step.agent_name} not found"
            )

        # Augment context with retrieved information
        augmented_context = {
            **(context or {}),
            "retrieved_info": retrieval_response.content
        }

        logger.info("Executing generation step with retrieved context")
        generation_response = await generation_agent.execute_task(
            initial_input,
            augmented_context
        )

        step_results.append({
            "agent": generation_step.agent_name,
            "action": "generate",
            "success": generation_response.success,
            "output": generation_response.content
        })

        outputs["generation"] = generation_response.content

        return WorkflowResult(
            success=generation_response.success,
            outputs=outputs,
            step_results=step_results,
            error=generation_response.error
        )

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get registered agent by name.

        Args:
            agent_name: Name of agent

        Returns:
            Agent instance or None
        """
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """
        List all registered agents.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def list_workflows(self) -> List[str]:
        """
        List all registered workflows.

        Returns:
            List of workflow names
        """
        return list(self.workflows.keys())
