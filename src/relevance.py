"""Graded relevance construction for incident-to-incident retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def compute_relevance_matrix(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> sparse.csr_matrix:
    """Compute graded relevance between test queries and train candidates.

    Relevance is 3 for matching CI and closure code, 2 for matching CI with a
    different closure code, 1 for matching CI subtype and closure code while
    using different CIs, and 0 otherwise. Comparisons are vectorized over
    query-candidate blocks to avoid nested Python loops.

    Args:
        df: Cleaned, time-sorted incident dataframe.
        train_indices: Row positions used as retrieval candidates.
        test_indices: Row positions used as retrieval queries.

    Returns:
        CSR matrix of shape ``(len(test_indices), len(train_indices))`` with
        int8 relevance values.
    """
    required = {"CI Name (aff)", "CI Subtype (aff)", "Closure Code"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    if train.ndim != 1 or test.ndim != 1:
        raise ValueError("train_indices and test_indices must be one-dimensional")

    train_values = df.iloc[train]
    test_values = df.iloc[test]
    train_ci = train_values["CI Name (aff)"].astype("string").fillna("<missing>").to_numpy()
    test_ci = test_values["CI Name (aff)"].astype("string").fillna("<missing>").to_numpy()
    train_subtype = train_values["CI Subtype (aff)"].astype("string").fillna("<missing>").to_numpy()
    test_subtype = test_values["CI Subtype (aff)"].astype("string").fillna("<missing>").to_numpy()
    train_closure = train_values["Closure Code"].astype("string").fillna("<missing>").to_numpy()
    test_closure = test_values["Closure Code"].astype("string").fillna("<missing>").to_numpy()

    same_ci = test_ci[:, None] == train_ci[None, :]
    same_subtype = test_subtype[:, None] == train_subtype[None, :]
    same_closure = test_closure[:, None] == train_closure[None, :]
    relevance = np.where(
        same_ci & same_closure,
        3,
        np.where(same_ci, 2, np.where(same_subtype & same_closure, 1, 0)),
    ).astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test), len(train)))
