"""Build target-free heterogeneous incident graphs for v2 experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData


_NODE_COLUMNS: dict[str, str] = {
    "ci": "CI Name (aff)",
    "ci_cby": "CI Name (CBy)",
    "ci_type": "CI Type (aff)",
    "ci_subtype": "CI Subtype (aff)",
    "category": "Category",
}

_EDGE_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    ("incident", "affects", "ci", "CI Name (aff)", "rev_affects"),
    ("incident", "caused_by", "ci_cby", "CI Name (CBy)", "rev_caused_by"),
    ("incident", "has_type", "ci_type", "CI Type (aff)", "rev_has_type"),
    (
        "incident",
        "has_subtype",
        "ci_subtype",
        "CI Subtype (aff)",
        "rev_has_subtype",
    ),
    ("incident", "categorized_as", "category", "Category", "rev_categorized_as"),
)


def _incident_texts(df: pd.DataFrame) -> list[str]:
    """Build robust incident text from available descriptive columns."""
    columns = [
        "CI Type (aff)",
        "CI Subtype (aff)",
        "Category",
        "Priority",
        "Impact",
        "Urgency",
        "Service Component WBS",
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


def _entity_mapping(values: pd.Series) -> dict[Any, int]:
    """Create a deterministic mapping for non-null entity values."""
    unique_values = sorted(values.dropna().unique().tolist(), key=str)
    return {value: index for index, value in enumerate(unique_values)}


def _make_edges(
    df: pd.DataFrame,
    column: str,
    target_mapping: Mapping[Any, int],
) -> torch.Tensor:
    """Create an incident-to-entity edge index for one dataframe column."""
    pairs = [
        (incident_index, target_mapping[value])
        for incident_index, value in enumerate(df[column])
        if pd.notna(value) and value in target_mapping
    ]
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def build_incident_graph_no_target(
    df: pd.DataFrame,
    text_model_name: str = "all-MiniLM-L6-v2",
    incident_features: np.ndarray | torch.Tensor | None = None,
) -> HeteroData:
    """Build a heterogeneous incident graph without closure-code targets.

    The graph intentionally contains no ``closure_code`` node type and no
    ``resolved_by`` relation. Closure codes are stored only in
    ``data["incident"].y``. Non-incident features are random placeholders;
    ``models_v2`` replaces them with learnable embeddings during its forward
    pass. Node and edge types whose source column has no non-null values are
    omitted entirely.

    Args:
        df: Time-sorted cleaned incident dataframe.
        text_model_name: SentenceTransformer model name or local path.
        incident_features: Optional precomputed incident feature matrix with
            one row per incident. If omitted, default feature computation is
            used unchanged.

    Returns:
        A CPU-resident PyG HeteroData object suitable for NeighborLoader.
    """
    if df.empty:
        raise ValueError("df must contain at least one incident")
    required = {"Closure Code", "CI Name (aff)"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    mappings: dict[str, dict[Any, int]] = {}
    for node_type, column in _NODE_COLUMNS.items():
        if column in df.columns:
            mapping = _entity_mapping(df[column])
            if mapping:
                mappings[node_type] = mapping

    closure_mapping = _entity_mapping(df["Closure Code"])
    if not closure_mapping:
        raise ValueError("Closure Code must contain at least one non-null value")

    if incident_features is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        encoder = SentenceTransformer(text_model_name, device=device)
        encoded = encoder.encode(
            _incident_texts(df),
            batch_size=256,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=False,
            device=device,
        )
        features = torch.as_tensor(encoded, dtype=torch.float32).cpu()
    else:
        features = torch.as_tensor(incident_features, dtype=torch.float32).cpu()
        if features.ndim != 2 or features.size(0) != len(df):
            raise ValueError(
                "incident_features must have shape (len(df), feature_dim)"
            )
    input_dim = features.size(1)

    data = HeteroData()
    data["incident"].x = features
    data["incident"].y = torch.tensor(
        [closure_mapping[value] for value in df["Closure Code"]],
        dtype=torch.long,
    )
    for node_type, mapping in mappings.items():
        data[node_type].x = torch.randn(len(mapping), input_dim)

    for source, relation, target, column, reverse_relation in _EDGE_SPECS:
        if target not in mappings or column not in df.columns:
            continue
        edge_index = _make_edges(df, column, mappings[target])
        if edge_index.numel() == 0:
            continue
        data[(source, relation, target)].edge_index = edge_index
        data[(target, reverse_relation, source)].edge_index = edge_index.flip(0)

    return data


def get_ci_node_names(
    data: HeteroData,
    df: pd.DataFrame,
    ci_col: str = "CI Name (aff)",
) -> list[str]:
    """Return CI names in graph node order."""
    if ci_col not in df.columns:
        raise KeyError(f"Missing required column: {ci_col}")
    mapping = _entity_mapping(df[ci_col])
    expected_count = data["ci"].num_nodes if "ci" in data.node_types else 0
    if len(mapping) != expected_count:
        raise ValueError("CI mapping does not match graph node count")
    return [str(name) for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def make_split_masks(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Create chronological boolean masks for train, validation, and test.

    Args:
        df: Already time-sorted incident dataframe.
        train_frac: Fraction assigned to the training prefix.
        val_frac: Fraction assigned to the validation segment.

    Returns:
        Dictionary containing boolean ``train_mask``, ``val_mask`` and
        ``test_mask`` tensors, each of length ``len(df)``.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1:
        raise ValueError("train_frac must be in (0, 1), val_frac in [0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")

    n_rows = len(df)
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    positions = torch.arange(n_rows)
    return {
        "train_mask": positions < train_end,
        "val_mask": (positions >= train_end) & (positions < val_end),
        "test_mask": positions >= val_end,
    }
