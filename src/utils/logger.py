import logging
import sys
from pathlib import Path
from config import get_settings


def setup_logger(name: str) -> logging.Logger:
    """
    Configure structured logging with consistent formatting.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured logger instance
    """
    settings = get_settings()

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with detailed formatting
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger for module."""
    return logging.getLogger(name)
