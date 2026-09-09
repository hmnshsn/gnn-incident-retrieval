"""Embed ServiceNow resolution text and cache vectors."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data_loader_sn import load_sn_incidents


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Embed filtered u_resolution values and save them as NumPy vectors."""
    set_seed(42)
    dataframe = load_sn_incidents()
    dataframe = dataframe[dataframe["Closure Code"].notna()].reset_index(drop=True)
    if "u_resolution" not in dataframe.columns:
        raise KeyError("Missing required column: u_resolution")

    resolutions = dataframe["u_resolution"]
    print(f"u_resolution null rate: {resolutions.isna().mean():.2%}")
    texts = resolutions.fillna("").astype(str).tolist()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)

    output_path = Path("results/sn_resolution_embeddings.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved: {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Non-null count: {resolutions.notna().sum()}")
    print(f"Mean/std: {embeddings.mean():.6f}/{embeddings.std():.6f}")
    for index in range(min(3, len(texts))):
        print(
            f"Example {index}: text={texts[index]!r}, "
            f"norm={np.linalg.norm(embeddings[index]):.6f}"
        )


if __name__ == "__main__":
    main()
