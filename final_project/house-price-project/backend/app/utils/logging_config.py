"""Logging configuration for the application."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quieten noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
