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
    definition: str = "D",
) -> sparse.csr_matrix:
    """Compute graded relevance between test queries and train candidates.

    Definition ``D`` is the default because relevance represents resolution
    transfer: it assigns 3 to matching closure and CI, 2 to matching closure
    and CI subtype, 1 to matching closure and category, and 0 otherwise.
    Definition ``A`` remains available for reproducing earlier results: it
    assigns 3 to matching CI and closure, 2 to matching CI, 1 to matching CI
    subtype and closure, and 0 otherwise. Null values never match; missing
    values are excluded from each criterion before comparisons to prevent
    placeholder values from creating false relevance. Comparisons are
    vectorized over query-candidate blocks to avoid nested Python loops.

    Args:
        df: Cleaned, time-sorted incident dataframe.
        train_indices: Row positions used as retrieval candidates.
        test_indices: Row positions used as retrieval queries.
        definition: Relevance definition, either ``"A"`` or ``"D"``.

    Returns:
        CSR matrix of shape ``(len(test_indices), len(train_indices))`` with
        int8 relevance values.
    """
    if definition not in {"A", "D"}:
        raise ValueError("definition must be either 'A' or 'D'")
    required = {"CI Name (aff)", "CI Subtype (aff)", "Closure Code"}
    if definition == "D":
        required.add("Category")
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    if train.ndim != 1 or test.ndim != 1:
        raise ValueError("train_indices and test_indices must be one-dimensional")

    train_values = df.iloc[train]
    test_values = df.iloc[test]
    train_ci_raw = train_values["CI Name (aff)"]
    test_ci_raw = test_values["CI Name (aff)"]
    train_subtype_raw = train_values["CI Subtype (aff)"]
    test_subtype_raw = test_values["CI Subtype (aff)"]
    train_closure_raw = train_values["Closure Code"]
    test_closure_raw = test_values["Closure Code"]
    train_ci_valid = ~train_ci_raw.isna().to_numpy()
    test_ci_valid = ~test_ci_raw.isna().to_numpy()
    train_subtype_valid = ~train_subtype_raw.isna().to_numpy()
    test_subtype_valid = ~test_subtype_raw.isna().to_numpy()
    train_closure_valid = ~train_closure_raw.isna().to_numpy()
    test_closure_valid = ~test_closure_raw.isna().to_numpy()

    train_ci = train_ci_raw.astype("string").fillna("<missing>").to_numpy()
    test_ci = test_ci_raw.astype("string").fillna("<missing>").to_numpy()
    train_subtype = train_subtype_raw.astype("string").fillna("<missing>").to_numpy()
    test_subtype = test_subtype_raw.astype("string").fillna("<missing>").to_numpy()
    train_closure = train_closure_raw.astype("string").fillna("<missing>").to_numpy()
    test_closure = test_closure_raw.astype("string").fillna("<missing>").to_numpy()
    if definition == "D":
        train_category_raw = train_values["Category"]
        test_category_raw = test_values["Category"]
        train_category_valid = ~train_category_raw.isna().to_numpy()
        test_category_valid = ~test_category_raw.isna().to_numpy()
        train_category = train_category_raw.astype("string").fillna("<missing>").to_numpy()
        test_category = test_category_raw.astype("string").fillna("<missing>").to_numpy()

    same_ci = (
        test_ci_valid[:, None]
        & train_ci_valid[None, :]
        & (test_ci[:, None] == train_ci[None, :])
    )
    same_subtype = (
        test_subtype_valid[:, None]
        & train_subtype_valid[None, :]
        & (test_subtype[:, None] == train_subtype[None, :])
    )
    same_closure = (
        test_closure_valid[:, None]
        & train_closure_valid[None, :]
        & (test_closure[:, None] == train_closure[None, :])
    )
    if definition == "A":
        relevance = np.where(
            same_ci & same_closure,
            3,
            np.where(same_ci, 2, np.where(same_subtype & same_closure, 1, 0)),
        )
    else:
        same_category = (
            test_category_valid[:, None]
            & train_category_valid[None, :]
            & (test_category[:, None] == train_category[None, :])
        )
        relevance = np.where(
            same_closure & same_ci,
            3,
            np.where(
                same_closure & same_subtype,
                2,
                np.where(same_closure & same_category, 1, 0),
            ),
        )
    relevance = relevance.astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test), len(train)))


def compute_relevance_matrix_e(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    definition: str = "E",
) -> sparse.csr_matrix:
    """Compute Definition E relevance without closure-code gating.

    Definition E assigns 3 to matching CI, 2 to matching CI subtype when CI
    differs, 1 to matching category when both CI and subtype differ, and 0
    otherwise. Null values never match.

    Args:
        df: Cleaned, time-sorted incident dataframe.
        train_indices: Row positions used as retrieval candidates.
        test_indices: Row positions used as retrieval queries.
        definition: Must be ``"E"``; retained for signature compatibility.

    Returns:
        CSR matrix of shape ``(len(test_indices), len(train_indices))`` with
        int8 relevance values.
    """
    if definition != "E":
        raise ValueError("definition must be 'E'")
    required = {"CI Name (aff)", "CI Subtype (aff)", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    if train.ndim != 1 or test.ndim != 1:
        raise ValueError("train_indices and test_indices must be one-dimensional")

    train_values = df.iloc[train]
    test_values = df.iloc[test]
    comparisons = []
    for column in ("CI Name (aff)", "CI Subtype (aff)", "Category"):
        train_raw = train_values[column]
        test_raw = test_values[column]
        train_valid = ~train_raw.isna().to_numpy()
        test_valid = ~test_raw.isna().to_numpy()
        train_strings = train_raw.astype("string").fillna("<missing>").to_numpy()
        test_strings = test_raw.astype("string").fillna("<missing>").to_numpy()
        comparisons.append(
            test_valid[:, None]
            & train_valid[None, :]
            & (test_strings[:, None] == train_strings[None, :])
        )

    same_ci, same_subtype, same_category = comparisons
    relevance = np.where(
        same_ci,
        3,
        np.where(same_subtype, 2, np.where(same_category, 1, 0)),
    ).astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test), len(train)))
