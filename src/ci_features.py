"""Build inductive CI features from incident metadata."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

_ATTRIBUTES = (
    "business_service",
    "service_offering",
    "assignment_group",
    "CI Subtype (aff)",
)


def build_ci_feature_matrix(
    df: pd.DataFrame,
    ci_col: str = "CI Name (aff)",
) -> tuple[np.ndarray, dict[str, int], int, dict[str, dict[str, object]]]:
    """Build inductive CI features from metadata modes and one-hot vocabularies."""
    required = {ci_col, *_ATTRIBUTES}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    ci_values = df[ci_col].dropna().unique().tolist()
    ci_values = sorted(ci_values, key=str)
    ci_to_idx = {str(value): index for index, value in enumerate(ci_values)}
    grouped = df.groupby(ci_col, sort=False, dropna=True)[list(_ATTRIBUTES)]
    modes = grouped.agg(
        lambda values: values.mode().iloc[0]
        if not values.dropna().empty
        else None
    )

    attribute_info: dict[str, dict[str, object]] = {}
    offset = 0
    for attribute in _ATTRIBUTES:
        categories = sorted(df[attribute].dropna().unique().tolist(), key=str)
        attribute_info[attribute] = {
            "categories": categories,
            "offset": offset,
            "dim": len(categories),
        }
        offset += len(categories)

    features = np.zeros((len(ci_values), offset), dtype=np.float32)
    coverage: dict[str, int] = {attribute: 0 for attribute in _ATTRIBUTES}
    category_indices = {
        attribute: {
            str(category): index
            for index, category in enumerate(
                attribute_info[attribute]["categories"]
            )
        }
        for attribute in _ATTRIBUTES
    }
    for ci_value, ci_index in ci_to_idx.items():
        mode_row = modes.loc[next(value for value in ci_values if str(value) == ci_value)]
        for attribute in _ATTRIBUTES:
            mode = mode_row[attribute]
            if pd.isna(mode):
                continue
            category_index = category_indices[attribute].get(str(mode))
            if category_index is not None:
                info = attribute_info[attribute]
                features[
                    ci_index,
                    int(info["offset"]) + category_index,
                ] = 1.0
                coverage[attribute] += 1

    zero_count = int(np.all(features == 0, axis=1).sum())
    logger.info("CI feature_dim=%d, all_zero_cis=%d", offset, zero_count)
    for attribute in _ATTRIBUTES:
        logger.info(
            "CI attribute coverage %s: %d/%d (%.1f%%)",
            attribute,
            coverage[attribute],
            len(ci_values),
            100 * coverage[attribute] / len(ci_values) if ci_values else 0.0,
        )
    return features, ci_to_idx, offset, attribute_info


def align_ci_features_to_graph(
    ci_features: np.ndarray,
    ci_to_idx: dict[str, int],
    graph_ci_names: list[str],
) -> np.ndarray:
    """Align CI feature rows with graph CI node ordering, zero-filling misses."""
    if ci_features.ndim != 2:
        raise ValueError("ci_features must be a two-dimensional array")
    aligned = np.zeros(
        (len(graph_ci_names), ci_features.shape[1]),
        dtype=ci_features.dtype,
    )
    for graph_index, ci_name in enumerate(graph_ci_names):
        source_index = ci_to_idx.get(str(ci_name))
        if source_index is not None:
            aligned[graph_index] = ci_features[source_index]
    return aligned
