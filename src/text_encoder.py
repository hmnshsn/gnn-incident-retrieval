"""Sentence-transformer encoding utilities for incident text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_cached_embeddings(cache_path: str | Path) -> np.ndarray:
    """Load incident embeddings from a NumPy cache file.

    Args:
        cache_path: Path to a ``.npy`` embeddings file.

    Returns:
        Cached embedding matrix.

    Raises:
        FileNotFoundError: If ``cache_path`` does not exist.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def encode_incidents(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: str | Path | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode concatenated incident descriptions with SentenceTransformer.

    ``short_description`` and ``description`` are joined with ``[SEP]`` when
    both are present. When only one is present, that field is used alone; when
    both are null, an empty string is encoded. Existing cache files are loaded
    without encoding, while newly generated embeddings are saved when a cache
    path is supplied.

    Args:
        df: Incident DataFrame containing ``short_description`` and
            ``description`` columns.
        model_name: SentenceTransformer model name or local path.
        cache_path: Optional ``.npy`` cache path.
        batch_size: Encoding batch size.

    Returns:
        NumPy array with shape ``(len(df), embedding_dim)``.
    """
    required = {"short_description", "description"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if cache_path is not None and Path(cache_path).exists():
        return load_cached_embeddings(cache_path)

    short_description = df["short_description"].fillna("").astype(str)
    description = df["description"].fillna("").astype(str)
    both_present = short_description.ne("") & description.ne("")
    texts = np.where(
        both_present,
        short_description + " [SEP] " + description,
        short_description.where(short_description.ne(""), description),
    ).tolist()

    print(f"Encoding {len(texts)} incidents with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings)
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
    print(f"Encoded embeddings shape: {embeddings.shape}")
    return embeddings
