"""Similarity baselines for incident-to-incident retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _column_values(
    df: pd.DataFrame,
    indices: Sequence[int],
    column: str,
) -> np.ndarray:
    """Return selected column values as comparable string arrays."""
    if column not in df.columns:
        raise KeyError(f"DataFrame must contain {column!r}")
    return (
        df.iloc[np.asarray(indices, dtype=int)][column]
        .astype("string")
        .fillna("<missing>")
        .to_numpy()
    )


def random_similarity(n_test: int, n_train: int) -> np.ndarray:
    """Generate reproducible random similarity scores.

    Args:
        n_test: Number of test queries.
        n_train: Number of training candidates.

    Returns:
        Dense float array of shape ``(n_test, n_train)``.
    """
    if n_test < 1 or n_train < 1:
        raise ValueError("n_test and n_train must be positive")
    return np.random.default_rng(42).random((n_test, n_train))


def ci_match_similarity(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> np.ndarray:
    """Score candidates one when CI matches, zero otherwise."""
    train_ci = _column_values(df, train_indices, "CI Name (aff)")
    test_ci = _column_values(df, test_indices, "CI Name (aff)")
    return (test_ci[:, None] == train_ci[None, :]).astype(np.float32)


def ci_subtype_similarity(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> np.ndarray:
    """Score same-CI candidates one and same-subtype candidates 0.5."""
    train_ci = _column_values(df, train_indices, "CI Name (aff)")
    test_ci = _column_values(df, test_indices, "CI Name (aff)")
    train_subtype = _column_values(df, train_indices, "CI Subtype (aff)")
    test_subtype = _column_values(df, test_indices, "CI Subtype (aff)")
    same_ci = test_ci[:, None] == train_ci[None, :]
    same_subtype = test_subtype[:, None] == train_subtype[None, :]
    return np.where(same_ci, 1.0, np.where(same_subtype, 0.5, 0.0)).astype(
        np.float32
    )


def category_similarity(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> np.ndarray:
    """Score candidates one when category matches, zero otherwise."""
    train_category = _column_values(df, train_indices, "Category")
    test_category = _column_values(df, test_indices, "Category")
    return (test_category[:, None] == train_category[None, :]).astype(np.float32)
