"""Evaluate BPI 2014 retrieval by CI visibility in training."""

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

from src.data_loader import load_bpi2014_incidents, temporal_split
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.graph_builder import build_incident_graph_no_target
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


def _set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_positions(
    n_rows: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-overlapping positional train, validation, and test indices."""
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    return (
        np.arange(train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n_rows),
    )


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding row, preserving zero rows."""
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


def _evaluate_group(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    """Evaluate one test CI-visibility group against all train candidates."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    similarities = _cosine_similarity(
        embeddings[test_indices],
        embeddings[train_indices],
    )
    return _metrics_for_similarity(similarities, relevance)


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
    """Train BPI GNN, evaluate CI cold-start groups, and save metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    logger = setup_logging("bpi_exp07_cold_start")
    _set_seed(args.seed)

    dataframe = load_bpi2014_incidents()
    train_df, val_df, test_df = temporal_split(dataframe)
    train_indices, val_indices, test_indices = _split_positions(len(dataframe))
    if (
        len(train_df) != len(train_indices)
        or len(val_df) != len(val_indices)
        or len(test_df) != len(test_indices)
    ):
        raise RuntimeError("temporal split lengths do not match positional indices")

    data = build_incident_graph_no_target(dataframe)
    node_counts = {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }
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
    model, history = train_minibatch(
        model,
        data,
        torch.as_tensor(train_indices, dtype=torch.long),
        torch.as_tensor(val_indices, dtype=torch.long),
        epochs=args.epochs,
        lr=0.005,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        device=str(device),
        patience=10,
    )

    full_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    embeddings = extract_incident_embeddings(model, data, full_loader, str(device)).numpy()
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]
    results = {
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
        "all": _evaluate_group(embeddings, dataframe, train_indices, test_indices),
    }
    logger.info("BPI cold-start summary:\n%s", _format_summary(results))

    output_path = Path("results/exp07_bpi_cold_start/cold_start_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset": "bpi2014",
                "best_epoch": history["best_epoch"],
                "stopped_early": history["stopped_early"],
                "num_test_seen": len(seen_test_indices),
                "num_test_unseen": len(unseen_test_indices),
                **results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
