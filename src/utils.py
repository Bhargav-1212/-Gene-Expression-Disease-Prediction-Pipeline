"""
utils.py
--------
Shared utilities: logging setup, config loading, and seed fixing.
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Any, Dict


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure root logger with console + optional file handler.

    Parameters
    ----------
    level    : Logging level string ("DEBUG", "INFO", "WARNING", …).
    log_file : Optional path to write logs to a file.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )
    # Suppress noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "numba", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Parse a YAML configuration file and return a nested dict.

    Parameters
    ----------
    config_path : Path to config.yaml.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_global_seed(seed: int = 42) -> None:
    """Fix Python, NumPy, and (if available) PyTorch random seeds."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def print_banner() -> None:
    """Print a startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║   Gene Expression Disease Prediction Pipeline  v1.0.0        ║
║   Modular ML System for Bioinformatics                       ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
