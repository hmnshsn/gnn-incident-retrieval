"""Shared logging setup for experiment runners."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(experiment_name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Create and return a console-and-file logger for one experiment.

    Args:
        experiment_name: Logger and log-file name prefix.
        log_dir: Directory where timestamped log files are written.

    Returns:
        Configured logger with INFO console and DEBUG file handlers.
    """
    for noisy_logger in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    )
    file_handler = logging.FileHandler(
        directory / f"{experiment_name}_{timestamp:%Y%m%d_%H%M%S}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return logger
