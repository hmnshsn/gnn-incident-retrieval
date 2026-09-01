"""Functional retrieval baselines for incident closure-code prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def _split_indices(split_indices: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract train and test row positions from common split representations."""
    if isinstance(split_indices, dict):
        train = split_indices.get("train", split_indices.get("train_indices"))
        test = split_indices.get("test", split_indices.get("test_indices"))
    elif isinstance(split_indices, Sequence) and not isinstance(split_indices, (str, bytes)):
        if len(split_indices) == 3:
            train, _, test = split_indices
        elif len(split_indices) == 2:
            train, test = split_indices
        else:
            raise ValueError("split_indices sequence must have length 2 or 3")
    else:
        raise TypeError("split_indices must be a mapping or sequence")
    if train is None or test is None:
        raise ValueError("split_indices must define train and test indices")
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def _closure_codes(df: pd.DataFrame) -> list[Any]:
    """Return deterministic closure-code candidates from the dataframe."""
    if "Closure Code" not in df.columns:
        raise KeyError("DataFrame must contain 'Closure Code'")
    return sorted(df["Closure Code"].dropna().unique().tolist(), key=str)


def _rank_for_scores(scores: Sequence[float], correct_index: int) -> int:
    """Return a one-indexed descending rank using stable tie handling."""
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return order.index(correct_index) + 1


def _target_index(value: Any, codes: Sequence[Any]) -> int:
    """Find a closure-code index and raise if the value is not a candidate."""
    try:
        return list(codes).index(value)
    except ValueError as exc:
        raise KeyError(f"Closure code {value!r} is not in the candidate set") from exc


def random_baseline(
    n_queries: int,
    n_candidates: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random one-indexed ranks as a lower-bound baseline.

    Args:
        n_queries: Number of queries to evaluate.
        n_candidates: Number of candidates ranked for each query.
        seed: Seed for reproducible sampling.

    Returns:
        LongTensor containing one random rank in ``[1, n_candidates]`` per query.
    """
    if n_queries < 1 or n_candidates < 1:
        raise ValueError("n_queries and n_candidates must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        1, n_candidates + 1, (n_queries,), generator=generator, dtype=torch.long
    )


def category_match_baseline(
    df: pd.DataFrame,
    split_indices: Any,
) -> torch.Tensor:
    """Rank closure codes by training frequency within each test category.

    If a test category was unseen during training, global training frequencies
    are used. Ties are resolved deterministically by closure-code order.

    Args:
        df: Cleaned incident DataFrame.
        split_indices: A ``(train, val, test)`` or ``(train, test)`` sequence,
            or a mapping containing ``train`` and ``test`` indices.

    Returns:
        LongTensor of one-indexed ranks for test incidents.
    """
    train_indices, test_indices = _split_indices(split_indices)
    codes = _closure_codes(df)
    if "Category" not in df.columns:
        raise KeyError("DataFrame must contain 'Category'")

    train_df = df.iloc[train_indices]
    global_counts = train_df["Closure Code"].value_counts()
    ranks: list[int] = []
    for test_index in test_indices:
        row = df.iloc[test_index]
        category_df = train_df[train_df["Category"] == row["Category"]]
        counts = category_df["Closure Code"].value_counts()
        if counts.empty:
            counts = global_counts
        scores = [float(counts.get(code, 0)) for code in codes]
        ranks.append(_rank_for_scores(scores, _target_index(row["Closure Code"], codes)))
    return torch.tensor(ranks, dtype=torch.long)


def ci_majority_baseline(
    df: pd.DataFrame,
    split_indices: Any,
) -> torch.Tensor:
    """Rank closure codes by training frequency for each incident CI.

    If a test incident's ``CI Name (aff)`` was not observed in training, the
    global training closure-code frequencies are used instead.

    Args:
        df: Cleaned incident DataFrame.
        split_indices: A ``(train, val, test)`` or ``(train, test)`` sequence,
            or a mapping containing ``train`` and ``test`` indices.

    Returns:
        LongTensor of one-indexed ranks for test incidents.
    """
    train_indices, test_indices = _split_indices(split_indices)
    if "CI Name (aff)" not in df.columns:
        raise KeyError("DataFrame must contain 'CI Name (aff)'")
    codes = _closure_codes(df)
    train_df = df.iloc[train_indices]
    global_counts = train_df["Closure Code"].value_counts()
    ranks: list[int] = []
    for test_index in test_indices:
        row = df.iloc[test_index]
        ci_rows = train_df[train_df["CI Name (aff)"] == row["CI Name (aff)"]]
        counts = ci_rows["Closure Code"].value_counts()
        if counts.empty:
            counts = global_counts
        scores = [float(counts.get(code, 0)) for code in codes]
        ranks.append(_rank_for_scores(scores, _target_index(row["Closure Code"], codes)))
    return torch.tensor(ranks, dtype=torch.long)


def _incident_texts(df: pd.DataFrame) -> list[str]:
    """Build robust text inputs from Summary or available incident fields."""
    if "Summary" in df.columns:
        return df["Summary"].fillna("no description").astype(str).tolist()
    columns = [
        "Category",
        "CI Name (aff)",
        "CI Type (aff)",
        "CI Subtype (aff)",
        "Status",
        "Priority",
        "KM number",
    ]
    available = [column for column in columns if column in df.columns]
    if not available:
        return ["no description"] * len(df)
    return (
        df[available]
        .fillna("no description")
        .astype(str)
        .agg(" | ".join, axis=1)
        .tolist()
    )


def text_similarity_baseline(
    df: pd.DataFrame,
    split_indices: Any,
    model_name: str = "all-MiniLM-L6-v2",
) -> torch.Tensor:
    """Rank closure codes by similarity to the most similar training incidents.

    Query and training texts are encoded on ``cuda:0`` when available. For each
    closure code, the mean of its top five query-to-training cosine similarities
    is used as the aggregate score. Similarity is evaluated in query batches to
    avoid materializing the full test-by-train matrix.

    Args:
        df: Cleaned incident DataFrame.
        split_indices: A ``(train, val, test)`` or ``(train, test)`` sequence,
            or a mapping containing ``train`` and ``test`` indices.
        model_name: SentenceTransformer model name or local path.

    Returns:
        LongTensor of one-indexed ranks for test incidents.
    """
    train_indices, test_indices = _split_indices(split_indices)
    codes = _closure_codes(df)
    texts = _incident_texts(df)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    train_embeddings = embeddings[torch.as_tensor(train_indices, device=device)]
    train_labels = df.iloc[train_indices]["Closure Code"].tolist()
    code_masks = {
        code: torch.tensor(
            [label == code for label in train_labels], dtype=torch.bool, device=device
        )
        for code in codes
    }

    ranks: list[int] = []
    for batch_start in range(0, len(test_indices), 256):
        batch_indices = test_indices[batch_start : batch_start + 256]
        query_embeddings = embeddings[torch.as_tensor(batch_indices, device=device)]
        similarities = query_embeddings @ train_embeddings.t()
        for row_offset, test_index in enumerate(batch_indices):
            scores: list[float] = []
            for code in codes:
                code_scores = similarities[row_offset][code_masks[code]]
                if code_scores.numel() == 0:
                    scores.append(float("-inf"))
                    continue
                top_k = min(5, code_scores.numel())
                scores.append(float(torch.topk(code_scores, top_k).values.mean().item()))
            correct_index = _target_index(df.iloc[test_index]["Closure Code"], codes)
            ranks.append(_rank_for_scores(scores, correct_index))
    return torch.tensor(ranks, dtype=torch.long)
