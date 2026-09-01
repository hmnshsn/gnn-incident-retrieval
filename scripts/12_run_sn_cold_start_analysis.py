"""Evaluate SN retrieval quality for seen and unseen configuration items."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data_loader_sn import load_sn_incidents
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.logging_config import setup_logging
from src.relevance import compute_relevance_matrix


RETRIEVAL_KS = (1, 5, 10, 20)
METHODS = ("text_only", "gnn_only", "late_fusion")


def _set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Evaluate similarities and return lowercase metric names."""
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
    gnn_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    dataframe: object,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> dict[str, dict[str, float]]:
    """Evaluate text, GNN, and late-fusion retrieval for one test group."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    gnn_test = gnn_embeddings[test_indices]
    gnn_train = gnn_embeddings[train_indices]
    text_test = text_embeddings[test_indices]
    text_train = text_embeddings[train_indices]
    gnn_similarity = _cosine_similarity(gnn_test, gnn_train)
    text_similarity = _cosine_similarity(text_test, text_train)
    gnn_test_normed = _normalize_rows(gnn_test)
    gnn_train_normed = _normalize_rows(gnn_train)
    text_test_normed = _normalize_rows(text_test)
    text_train_normed = _normalize_rows(text_train)
    if alpha == 1.0:
        fusion_similarity = gnn_similarity
    elif alpha == 0.0:
        fusion_similarity = text_similarity
    else:
        fused_test = np.concatenate(
            [alpha * gnn_test_normed, (1.0 - alpha) * text_test_normed],
            axis=1,
        )
        fused_train = np.concatenate(
            [alpha * gnn_train_normed, (1.0 - alpha) * text_train_normed],
            axis=1,
        )
        fusion_similarity = _cosine_similarity(fused_test, fused_train)
    return {
        "text_only": _metrics_for_similarity(text_similarity, relevance),
        "gnn_only": _metrics_for_similarity(gnn_similarity, relevance),
        "late_fusion": _metrics_for_similarity(fusion_similarity, relevance),
    }


def _format_summary(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Format seen, unseen, and all-group metrics as a summary table."""
    lines = [
        "Group   Method       nDCG@1  nDCG@5  nDCG@10 nDCG@20 MAP     MRR",
    ]
    for group in ("seen", "unseen", "all"):
        for method in METHODS:
            metrics = results[group][method]
            lines.append(
                f"{group:<7} {method:<12}"
                f"{metrics['ndcg@1']:.3f}   "
                f"{metrics['ndcg@5']:.3f}   "
                f"{metrics['ndcg@10']:.3f}   "
                f"{metrics['ndcg@20']:.3f}   "
                f"{metrics['map']:.3f}   "
                f"{metrics['mrr']:.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    """Run cold-start retrieval analysis and save grouped metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    logger = setup_logging("sn_exp04_cold_start")
    _set_seed(args.seed)

    gnn_embeddings = np.load("results/sn_gnn_embeddings.npy")
    text_embeddings_full = np.load("results/sn_text_embeddings.npy")
    split_data = np.load("results/sn_split_indices.npz")
    train_indices = split_data["train_indices"].astype(np.int64)
    test_indices = split_data["test_indices"].astype(np.int64)
    graph_row_indices = split_data["graph_row_indices"].astype(np.int64)
    text_embeddings = text_embeddings_full[graph_row_indices]

    dataframe = load_sn_incidents()
    dataframe = dataframe.dropna(subset=["Closure Code"]).reset_index(drop=True)
    if len(dataframe) != len(gnn_embeddings) or len(dataframe) != len(text_embeddings):
        raise ValueError("aligned embedding row counts do not match SN incident data")

    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_indices = test_indices[seen_mask.to_numpy()]
    unseen_indices = test_indices[~seen_mask.to_numpy()]

    results = {
        "seen": _evaluate_group(
            gnn_embeddings,
            text_embeddings,
            dataframe,
            train_indices,
            seen_indices,
            args.alpha,
        ),
        "unseen": _evaluate_group(
            gnn_embeddings,
            text_embeddings,
            dataframe,
            train_indices,
            unseen_indices,
            args.alpha,
        ),
        "all": _evaluate_group(
            gnn_embeddings,
            text_embeddings,
            dataframe,
            train_indices,
            test_indices,
            args.alpha,
        ),
    }
    logger.info("Cold-start summary:\n%s", _format_summary(results))

    output_path = Path("results/exp04_sn_cold_start/cold_start_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "num_test_seen": len(seen_indices),
                "num_test_unseen": len(unseen_indices),
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
