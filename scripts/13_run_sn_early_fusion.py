"""Run SN early-fusion feature ablations for incident retrieval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.data_loader import temporal_split
from src.data_loader_sn import load_sn_incidents
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.impute_subcategory import impute_subcategory
from src.logging_config import setup_logging
from src.models import (
    HeteroIncidentClassifier,
    extract_incident_embeddings,
    train_minibatch,
)
from src.relevance import compute_relevance_matrix


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
RETRIEVAL_KS = (1, 5, 10, 20)


def _set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_dataframe(
    subtype_mode: str,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Load SN data, apply subtype mode, and track graph-row positions."""
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


def _split_positions(
    n_rows: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positional train, validation, and test indices."""
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    return (
        np.arange(train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n_rows),
    )


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows, preserving zero rows."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings, dtype=np.float32),
        where=norms != 0,
    )


def _cosine_similarity(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query and candidate embeddings."""
    return _normalize_rows(query_embeddings) @ _normalize_rows(candidate_embeddings).T


def _metrics_for_similarity(
    similarities: np.ndarray,
    relevance: object,
) -> dict[str, float]:
    """Evaluate similarity scores with lowercase metric keys."""
    metrics = evaluate_retrieval_graded(
        similarities,
        relevance,
        ks=RETRIEVAL_KS,
    )
    return {
        "ndcg@1": metrics["nDCG@1"],
        "ndcg@5": metrics["nDCG@5"],
        "ndcg@10": metrics["nDCG@10"],
        "ndcg@20": metrics["nDCG@20"],
        "map": metrics["MAP"],
        "mrr": metrics["MRR"],
    }


def _build_node_counts(data: HeteroData) -> dict[str, int]:
    """Extract positive non-incident node counts from a graph."""
    return {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }


def _evaluate_group(
    gnn_embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    """Evaluate retrieval using trained early-fusion GNN embeddings."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    gnn_test = gnn_embeddings[test_indices]
    gnn_train = gnn_embeddings[train_indices]
    return _metrics_for_similarity(
        _cosine_similarity(gnn_test, gnn_train),
        relevance,
    )


def _format_summary(results: dict[str, dict[str, float]]) -> str:
    """Format seen, unseen, and all retrieval metrics."""
    lines = ["Group   nDCG@1  nDCG@5  nDCG@10 nDCG@20 MAP     MRR"]
    for group in ("seen", "unseen", "all"):
        metrics = results[group]
        lines.append(
            f"{group:<7}"
            f"{metrics['ndcg@1']:.3f}   "
            f"{metrics['ndcg@5']:.3f}   "
            f"{metrics['ndcg@10']:.3f}   "
            f"{metrics['ndcg@20']:.3f}   "
            f"{metrics['map']:.3f}   "
            f"{metrics['mrr']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Train one early-fusion mode and save grouped retrieval metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        choices=["tfidf", "text", "concat"],
        default="text",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--subtype",
        choices=["raw", "imputed", "none"],
        default="raw",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    logger = setup_logging(f"sn_exp05_early_fusion_{args.features}")
    _set_seed(args.seed)

    dataframe, splits, graph_row_indices = _prepare_dataframe(args.subtype)
    train_df, val_df, test_df = splits
    train_indices, _, test_indices = _split_positions(len(dataframe))
    text_embeddings_full = np.load("results/sn_text_embeddings.npy")
    text_embeddings = text_embeddings_full[graph_row_indices]
    data = build_incident_graph_no_target(
        dataframe,
        incident_features=text_embeddings if args.features == "text" else None,
    )
    if args.features == "concat":
        base_features = data["incident"].x.numpy()
        if base_features.shape[0] != text_embeddings.shape[0]:
            raise ValueError("base and text feature row counts do not match")
        concatenated = np.concatenate([base_features, text_embeddings], axis=1)
        data["incident"].x = torch.as_tensor(concatenated, dtype=torch.float32)

    masks = make_split_masks(dataframe)
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
    embeddings = extract_incident_embeddings(model, data, full_loader, str(device)).numpy()
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]
    group_results = {
        "seen": _evaluate_group(
            embeddings,
            dataframe,
            train_indices,
            seen_test_indices,
        ),
        "unseen": _evaluate_group(
            embeddings,
            dataframe,
            train_indices,
            unseen_test_indices,
        ),
        "all": _evaluate_group(
            embeddings,
            dataframe,
            train_indices,
            test_indices,
        ),
    }
    logger.info("Early-fusion summary:\n%s", _format_summary(group_results))

    output_path = Path(f"results/exp05_sn_early_fusion/early_fusion_{args.features}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "features": args.features,
                "epochs": args.epochs,
                "subtype": args.subtype,
                "num_test_seen": len(seen_test_indices),
                "num_test_unseen": len(unseen_test_indices),
                **group_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
