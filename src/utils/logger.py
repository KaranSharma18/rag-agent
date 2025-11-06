"""
Logging configuration for detailed agent operation tracking.
"""
import logging
import sys
from typing import Optional
from .config import LOG_LEVEL, LOG_FORMAT, LOG_TO_FILE, LOG_FILE


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with detailed formatting.
    
    Args:
        name: Logger name (usually __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter with colors for console (optional, but helpful)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if LOG_TO_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_agent_step(logger: logging.Logger, step: str, details: dict):
    """
    Log an agent step with structured details.
    
    Args:
        logger: Logger instance
        step: Name of the step (e.g., "REASONING", "RETRIEVAL")
        details: Dictionary of step details
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"AGENT STEP: {step}")
    logger.info(f"{'='*60}")
    for key, value in details.items():
        if isinstance(value, (list, dict)):
            logger.info(f"{key}:")
            if isinstance(value, list):
                for item in value[:3]:  # Show first 3 items
                    logger.info(f"  - {str(item)[:200]}")  # Truncate long strings
                if len(value) > 3:
                    logger.info(f"  ... and {len(value) - 3} more")
            else:
                for k, v in value.items():
                    logger.info(f"  {k}: {str(v)[:200]}")
        else:
            logger.info(f"{key}: {str(value)[:500]}")
    logger.info(f"{'='*60}\n")


def log_retrieval_results(logger: logging.Logger, query: str, results: list):
    """
    Log retrieval results in a readable format.
    
    Args:
        logger: Logger instance
        query: Search query
        results: List of retrieved documents
    """
    logger.info(f"\n{'*'*60}")
    logger.info(f"RETRIEVAL RESULTS")
    logger.info(f"Query: {query}")
    logger.info(f"Found: {len(results)} documents")
    logger.info(f"{'*'*60}")
    
    for idx, doc in enumerate(results, 1):
        logger.info(f"\nDocument {idx}:")
        logger.info(f"  Content: {doc.get('content', '')[:200]}...")
        logger.info(f"  Metadata: {doc.get('metadata', {})}")
    
    logger.info(f"{'*'*60}\n")
