"""Train SN GNN with inductive CI features and evaluate cold-start retrieval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.ci_features import align_ci_features_to_graph, build_ci_feature_matrix
from src.data_loader import temporal_split
from src.data_loader_sn import load_sn_incidents
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.graph_builder import (
    build_incident_graph_no_target,
    get_ci_node_names,
    make_split_masks,
)
from src.impute_subcategory import impute_subcategory
from src.logging_config import setup_logging
from src.models import HeteroIncidentClassifier, extract_incident_embeddings, train_minibatch
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
        dataframe = impute_subcategory(dataframe, int(len(dataframe) * 0.7))
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
    metrics = evaluate_retrieval_graded(similarities, relevance, ks=RETRIEVAL_KS)
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
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    """Evaluate one test CI-visibility group against all train candidates."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    similarities = _cosine_similarity(
        embeddings[test_indices], embeddings[train_indices]
    )
    return _metrics_for_similarity(similarities, relevance)


def _format_summary(results: dict[str, dict[str, float]]) -> str:
    """Format seen, unseen, and all retrieval metrics."""
    lines = ["Group   nDCG@1  nDCG@5  nDCG@10 nDCG@20 MAP     MRR"]
    for group in ("seen", "unseen", "all"):
        metrics = results[group]
        lines.append(
            f"{group:<7}{metrics['ndcg@1']:.3f}   "
            f"{metrics['ndcg@5']:.3f}   {metrics['ndcg@10']:.3f}   "
            f"{metrics['ndcg@20']:.3f}   {metrics['map']:.3f}   "
            f"{metrics['mrr']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Train inductive CI model and save grouped retrieval metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-features", choices=["learnable", "inductive", "both"], default="inductive"
    )
    parser.add_argument("--ci-dropout-rate", type=float, default=0.0)
    parser.add_argument(
        "--ci-dropout-mode",
        choices=["embedding", "edge", "both"],
        default="embedding",
    )
    parser.add_argument("--features", choices=["tfidf", "text", "concat"], default="tfidf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.ci_dropout_rate <= 1.0:
        raise ValueError("ci-dropout-rate must be between 0.0 and 1.0")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    logger = setup_logging("sn_exp09_inductive_ci")
    _set_seed(args.seed)

    dataframe, splits, graph_row_indices = _prepare_dataframe(args.subtype)
    train_df, val_df, test_df = splits
    train_indices, _, test_indices = _split_positions(len(dataframe))
    masks = make_split_masks(dataframe)

    text_embeddings = None
    if args.features in {"text", "concat"}:
        text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    if args.features == "text":
        data = build_incident_graph_no_target(dataframe, incident_features=text_embeddings)
    else:
        data = build_incident_graph_no_target(dataframe)
        if args.features == "concat":
            base_features = data["incident"].x.numpy()
            if text_embeddings is None or base_features.shape[0] != text_embeddings.shape[0]:
                raise ValueError("base and text feature row counts do not match")
            data["incident"].x = torch.as_tensor(
                np.concatenate([base_features, text_embeddings], axis=1), dtype=torch.float32
            )

    train_ci_features, ci_to_idx, feature_dim, _ = build_ci_feature_matrix(train_df)
    graph_ci_names = get_ci_node_names(data, dataframe)
    aligned_features = align_ci_features_to_graph(
        train_ci_features, ci_to_idx, graph_ci_names
    )
    ci_input_features = torch.as_tensor(aligned_features, dtype=torch.float32)
    model_feature_dim = feature_dim if args.ci_features != "learnable" else None
    model = HeteroIncidentClassifier(
        data.metadata(),
        node_counts=_build_node_counts(data),
        input_dim=data["incident"].x.size(1),
        hidden_dim=128,
        num_classes=int(dataframe["Closure Code"].nunique()),
        num_layers=2,
        dropout=0.3,
        ci_dropout_rate=args.ci_dropout_rate,
        ci_dropout_mode=args.ci_dropout_mode,
        ci_feature_dim=model_feature_dim,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    logger.info(
        "Loaders configured: full_batches=%d, train=%d, val=%d, test=%d",
        len(full_loader), len(train_df), len(val_df), len(test_df)
    )
    training_features = ci_input_features if args.ci_features != "learnable" else None
    model, history = train_minibatch(
        model,
        data,
        masks["train_mask"],
        masks["val_mask"],
        epochs=args.epochs,
        lr=0.005,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        device=str(device),
        patience=args.patience,
        ci_input_features=training_features,
        ci_features_mode=args.ci_features,
    )
    embeddings = extract_incident_embeddings(
        model,
        data,
        full_loader,
        str(device),
        training_features,
        args.ci_features,
    ).numpy()
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]
    results = {
        "seen": _evaluate_group(embeddings, dataframe, train_indices, seen_test_indices),
        "unseen": _evaluate_group(embeddings, dataframe, train_indices, unseen_test_indices),
        "all": _evaluate_group(embeddings, dataframe, train_indices, test_indices),
    }
    logger.info("Inductive CI summary:\n%s", _format_summary(results))
    output_path = Path(
        f"results/exp09_sn_inductive_ci/inductive_ci_"
        f"{args.ci_features}_{args.features}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "ci_features_mode": args.ci_features,
                "features": args.features,
                "ci_feature_dim": feature_dim,
                "ci_dropout_rate": args.ci_dropout_rate,
                "ci_dropout_mode": args.ci_dropout_mode,
                "epochs_trained": len(history["train_loss"]),
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
