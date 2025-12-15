"""Configuration management for LLM workflow orchestration."""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    type: str = "chromadb"
    collection_name: str = "default"
    persist_directory: str = "./vector_db"
    host: str = "localhost"
    port: int = 6333
    api_key: str = ""
    environment: str = ""
    dimension: int = 768
    metric: str = "cosine"


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str = "default_agent"
    role: str = "generator"
    system_prompt: str = ""
    llm_config: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    name: str = "default_workflow"
    type: str = "sequential"
    agents: list = field(default_factory=list)
    max_iterations: int = 10


@dataclass
class Config:
    """Main configuration class."""
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    workflows: Dict[str, WorkflowConfig] = field(default_factory=dict)
    logging_level: str = "INFO"
    log_file: str = "llm_workflow.log"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file and environment variables.

    Args:
        config_path: Path to configuration file (YAML or JSON)

    Returns:
        Configuration object
    """
    # Load environment variables
    load_dotenv()

    config = Config()

    # If config file provided, load it
    if config_path and os.path.exists(config_path):
        config = _load_from_file(config_path)

    # Override with environment variables
    config = _apply_env_overrides(config)

    logger.info(f"Loaded configuration")
    return config


def _load_from_file(config_path: str) -> Config:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)

    try:
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        logger.info(f"Loaded config from {config_path}")
        return _dict_to_config(data)

    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return Config()


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object."""
    config = Config()

    # Vector DB config
    if 'vector_db' in data:
        vdb = data['vector_db']
        config.vector_db = VectorDBConfig(
            type=vdb.get('type', 'chromadb'),
            collection_name=vdb.get('collection_name', 'default'),
            persist_directory=vdb.get('persist_directory', './vector_db'),
            host=vdb.get('host', 'localhost'),
            port=vdb.get('port', 6333),
            api_key=vdb.get('api_key', ''),
            environment=vdb.get('environment', ''),
            dimension=vdb.get('dimension', 768),
            metric=vdb.get('metric', 'cosine')
        )

    # LLM config
    if 'llm' in data:
        llm = data['llm']
        config.llm = LLMConfig(
            provider=llm.get('provider', 'openai'),
            model_name=llm.get('model_name', 'gpt-3.5-turbo'),
            api_key=llm.get('api_key', ''),
            temperature=llm.get('temperature', 0.7),
            max_tokens=llm.get('max_tokens', 2000),
            timeout=llm.get('timeout', 30)
        )

    # Agents config
    if 'agents' in data:
        for agent_name, agent_data in data['agents'].items():
            agent_llm = LLMConfig(
                **agent_data.get('llm_config', {})
            ) if 'llm_config' in agent_data else config.llm

            config.agents[agent_name] = AgentConfig(
                name=agent_name,
                role=agent_data.get('role', 'generator'),
                system_prompt=agent_data.get('system_prompt', ''),
                llm_config=agent_llm
            )

    # Workflows config
    if 'workflows' in data:
        for workflow_name, workflow_data in data['workflows'].items():
            config.workflows[workflow_name] = WorkflowConfig(
                name=workflow_name,
                type=workflow_data.get('type', 'sequential'),
                agents=workflow_data.get('agents', []),
                max_iterations=workflow_data.get('max_iterations', 10)
            )

    # Logging config
    config.logging_level = data.get('logging_level', 'INFO')
    config.log_file = data.get('log_file', 'llm_workflow.log')

    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""

    # Vector DB overrides
    if os.getenv('VECTOR_DB_TYPE'):
        config.vector_db.type = os.getenv('VECTOR_DB_TYPE')
    if os.getenv('VECTOR_DB_COLLECTION'):
        config.vector_db.collection_name = os.getenv('VECTOR_DB_COLLECTION')
    if os.getenv('VECTOR_DB_API_KEY'):
        config.vector_db.api_key = os.getenv('VECTOR_DB_API_KEY')

    # LLM overrides
    if os.getenv('LLM_PROVIDER'):
        config.llm.provider = os.getenv('LLM_PROVIDER')
    if os.getenv('LLM_MODEL'):
        config.llm.model_name = os.getenv('LLM_MODEL')
    if os.getenv('LLM_API_KEY'):
        config.llm.api_key = os.getenv('LLM_API_KEY')
    if os.getenv('OPENAI_API_KEY'):
        config.llm.api_key = os.getenv('OPENAI_API_KEY')
    if os.getenv('ANTHROPIC_API_KEY'):
        config.llm.api_key = os.getenv('ANTHROPIC_API_KEY')

    # Logging overrides
    if os.getenv('LOG_LEVEL'):
        config.logging_level = os.getenv('LOG_LEVEL')

    return config


def save_config(config: Config, config_path: str):
    """
    Save configuration to file.

    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    path = Path(config_path)

    # Convert config to dict
    config_dict = {
        'vector_db': {
            'type': config.vector_db.type,
            'collection_name': config.vector_db.collection_name,
            'persist_directory': config.vector_db.persist_directory,
            'host': config.vector_db.host,
            'port': config.vector_db.port,
            'dimension': config.vector_db.dimension,
            'metric': config.vector_db.metric
        },
        'llm': {
            'provider': config.llm.provider,
            'model_name': config.llm.model_name,
            'temperature': config.llm.temperature,
            'max_tokens': config.llm.max_tokens,
            'timeout': config.llm.timeout
        },
        'logging_level': config.logging_level,
        'log_file': config.log_file
    }

    # Save based on extension
    try:
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        logger.info(f"Saved config to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config: {e}")
