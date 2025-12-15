"""Langflow component for agent orchestration."""

from langflow import CustomComponent
from typing import Dict, Any
import asyncio
import json

from ..agents import AgentOrchestrator, WorkflowConfig, WorkflowType, WorkflowStep


class OrchestratorComponent(CustomComponent):
    """Custom Langflow component for orchestrating multi-agent workflows."""

    display_name = "Agent Orchestrator"
    description = "Orchestrate multiple agents in complex workflows"
    documentation = "https://github.com/yourusername/llm-workflow-orchestration"

    def build_config(self) -> Dict[str, Any]:
        """Build component configuration."""
        return {
            "workflow_type": {
                "display_name": "Workflow Type",
                "options": ["sequential", "parallel", "rag"],
                "value": "sequential",
            },
            "workflow_name": {
                "display_name": "Workflow Name",
                "value": "default_workflow",
            },
            "workflow_config": {
                "display_name": "Workflow Configuration (JSON)",
                "multiline": True,
                "value": json.dumps({
                    "steps": [
                        {
                            "agent_name": "agent1",
                            "action": "process",
                            "inputs": {}
                        }
                    ]
                }, indent=2),
            },
            "initial_input": {
                "display_name": "Initial Input",
                "multiline": True,
                "value": "",
            },
            "context": {
                "display_name": "Context (JSON)",
                "multiline": True,
                "value": "{}",
            },
        }

    def build(
        self,
        workflow_type: str,
        workflow_name: str,
        workflow_config: str,
        initial_input: str,
        context: str = "{}",
    ) -> Dict[str, Any]:
        """
        Execute orchestrated workflow.

        Args:
            workflow_type: Type of workflow
            workflow_name: Name of the workflow
            workflow_config: JSON configuration for workflow steps
            initial_input: Initial input to the workflow
            context: Optional context as JSON

        Returns:
            Workflow execution result
        """
        try:
            # Parse workflow config
            config_dict = json.loads(workflow_config)
            context_dict = json.loads(context)

            # Map workflow type
            type_map = {
                "sequential": WorkflowType.SEQUENTIAL,
                "parallel": WorkflowType.PARALLEL,
                "rag": WorkflowType.RAG,
            }
            wf_type = type_map.get(workflow_type, WorkflowType.SEQUENTIAL)

            # Create workflow steps
            steps = []
            for step_config in config_dict.get("steps", []):
                step = WorkflowStep(
                    agent_name=step_config.get("agent_name", ""),
                    action=step_config.get("action", "process"),
                    inputs=step_config.get("inputs", {}),
                    next_steps=step_config.get("next_steps", [])
                )
                steps.append(step)

            # Create workflow configuration
            workflow = WorkflowConfig(
                name=workflow_name,
                workflow_type=wf_type,
                steps=steps,
                initial_step=steps[0].agent_name if steps else ""
            )

            # Initialize orchestrator
            orchestrator = AgentOrchestrator()

            # Note: In real implementation, you would need to register agents
            # This is a simplified version
            orchestrator.register_workflow(workflow)

            # Execute workflow
            result = asyncio.run(
                orchestrator.execute_workflow(
                    workflow_name=workflow_name,
                    initial_input=initial_input,
                    context=context_dict
                )
            )

            return {
                "success": result.success,
                "outputs": result.outputs,
                "step_results": result.step_results,
                "error": result.error,
                "metadata": result.metadata
            }

        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON configuration: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
