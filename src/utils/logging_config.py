"""Logging configuration for LLM workflow orchestration."""

import sys
from loguru import logger
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "llm_workflow.log",
    rotation: str = "10 MB",
    retention: str = "1 week"
):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        rotation: Log rotation policy
        retention: Log retention policy
    """
    # Remove default logger
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # Add file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )

    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
