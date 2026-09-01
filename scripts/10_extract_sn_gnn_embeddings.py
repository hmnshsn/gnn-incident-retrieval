"""Train SN GNN and save incident embeddings and split indices."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from src.data_loader import temporal_split
from src.data_loader_sn import load_sn_incidents
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.impute_subcategory import impute_subcategory
from src.logging_config import setup_logging
from src.models import (
    HeteroIncidentClassifier,
    extract_incident_embeddings,
    train_minibatch,
)


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]


def _set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_dataframe(
    subtype_mode: str,
) -> tuple[object, tuple[object, object, object], np.ndarray]:
    """Load SN incidents, apply subtype mode, and create temporal splits."""
    dataframe = load_sn_incidents()
    original_indices = np.arange(len(dataframe), dtype=np.int64)
    if subtype_mode == "imputed":
        train_end = int(len(dataframe) * 0.7)
        dataframe = impute_subcategory(dataframe, train_end)
    elif subtype_mode == "none":
        dataframe["CI Subtype (aff)"] = np.nan
    valid_rows = dataframe["Closure Code"].notna()
    graph_row_indices = original_indices[valid_rows.to_numpy()]
    dataframe = dataframe.loc[valid_rows].reset_index(drop=True)
    splits = temporal_split(dataframe, time_col="opened_at")
    return dataframe, splits, graph_row_indices


def _build_node_counts(data: object) -> dict[str, int]:
    """Extract positive non-incident node counts from a graph."""
    return {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }


def main() -> None:
    """Train SN GNN and save full-graph embeddings plus split indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--subtype",
        choices=["raw", "imputed", "none"],
        default="raw",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = setup_logging(f"sn_extract_gnn_embeddings_{args.subtype}")
    _set_seed(args.seed)

    dataframe, splits, graph_row_indices = _prepare_dataframe(args.subtype)
    train_df, val_df, test_df = splits
    masks = make_split_masks(dataframe)
    train_indices = torch.where(masks["train_mask"])[0].numpy()
    test_indices = torch.where(masks["test_mask"])[0].numpy()
    data = build_incident_graph_no_target(dataframe)
    node_counts = _build_node_counts(data)
    model = HeteroIncidentClassifier(
        data.metadata(),
        node_counts=node_counts,
        input_dim=data["incident"].x.size(1),
        hidden_dim=128,
        num_classes=int(dataframe["Closure Code"].nunique()),
        num_layers=2,
        dropout=0.3,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", masks["train_mask"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    full_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    logger.info(
        "Loaders configured: train_batches=%d, full_batches=%d, train=%d, val=%d, test=%d",
        len(train_loader),
        len(full_loader),
        len(train_df),
        len(val_df),
        len(test_df),
    )
    model, _ = train_minibatch(
        model,
        data,
        masks["train_mask"],
        masks["val_mask"],
        epochs=args.epochs,
        lr=0.005,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        device=str(device),
    )
    embeddings = extract_incident_embeddings(model, data, full_loader, str(device))

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = results_dir / "sn_gnn_embeddings.npy"
    splits_path = results_dir / "sn_split_indices.npz"
    np.save(embeddings_path, embeddings.numpy())
    np.savez(
        splits_path,
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        graph_row_indices=graph_row_indices.astype(np.int64),
    )
    logger.info("Saved embeddings: path=%s, shape=%s", embeddings_path, tuple(embeddings.shape))
    logger.info(
        "Saved split indices: path=%s, train=%d, test=%d",
        splits_path,
        len(train_indices),
        len(test_indices),
    )


if __name__ == "__main__":
    main()
