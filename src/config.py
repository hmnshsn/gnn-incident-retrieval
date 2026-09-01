"""Experiment hyperparameters and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Top-level experiment configuration."""

    # Paths
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("outputs")

    # Data split
    train_frac: float = 0.7
    val_frac: float = 0.15

    # Retrieval evaluation
    ks: tuple[int, ...] = (1, 3, 5, 10)

    # Sentence-transformer embedding model
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384

    # GNN
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 100
    device: str = "cuda"

    # Reproducibility
    seed: int = 42


DEFAULT_CONFIG: Config = Config()
