"""Precompute ServiceNow incident text embeddings."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.data_loader_sn import load_sn_incidents
from src.text_encoder import encode_incidents


def main() -> None:
    """Load ServiceNow incidents and cache SentenceTransformer embeddings."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    incidents = load_sn_incidents()
    embeddings = encode_incidents(
        incidents,
        model_name="all-MiniLM-L6-v2",
        cache_path="results/sn_text_embeddings.npy",
    )
    print(f"Saved embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
