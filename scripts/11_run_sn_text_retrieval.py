"""Evaluate text, GNN, and late-fusion SN retrieval embeddings."""

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
    query_normed = _normalize_rows(query_embeddings)
    candidate_normed = _normalize_rows(candidate_embeddings)
    return query_normed @ candidate_normed.T


def _metrics_for_similarity(
    similarities: np.ndarray,
    relevance: object,
) -> dict[str, float]:
    """Evaluate similarities and return the requested lowercase metric keys."""
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


def _format_metrics(name: str, metrics: dict[str, float]) -> str:
    """Format one retrieval result row for logging."""
    return (
        f"{name}: ndcg@1={metrics['ndcg@1']:.3f}, "
        f"ndcg@5={metrics['ndcg@5']:.3f}, "
        f"ndcg@10={metrics['ndcg@10']:.3f}, "
        f"ndcg@20={metrics['ndcg@20']:.3f}, "
        f"map={metrics['map']:.3f}, mrr={metrics['mrr']:.3f}"
    )


def main() -> None:
    """Load cached embeddings, evaluate three retrieval settings, and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    logger = setup_logging("sn_exp03_text_retrieval")
    _set_seed(args.seed)

    gnn_embeddings = np.load("results/sn_gnn_embeddings.npy")
    text_embeddings_full = np.load("results/sn_text_embeddings.npy")
    split_data = np.load("results/sn_split_indices.npz")
    train_indices = split_data["train_indices"].astype(np.int64)
    test_indices = split_data["test_indices"].astype(np.int64)
    graph_row_indices = split_data["graph_row_indices"].astype(np.int64)
    text_embeddings = text_embeddings_full[graph_row_indices]
    if text_embeddings.shape[0] != gnn_embeddings.shape[0]:
        raise ValueError("aligned text and GNN embedding row counts do not match")
    dataframe = load_sn_incidents()
    dataframe = dataframe.dropna(subset=["Closure Code"]).reset_index(drop=True)
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)

    gnn_test = gnn_embeddings[test_indices]
    gnn_train = gnn_embeddings[train_indices]
    text_test = text_embeddings[test_indices]
    text_train = text_embeddings[train_indices]
    gnn_similarity = _cosine_similarity(gnn_test, gnn_train)
    text_similarity = _cosine_similarity(text_test, text_train)
    gnn_normed_test = _normalize_rows(gnn_test)
    gnn_normed_train = _normalize_rows(gnn_train)
    text_normed_test = _normalize_rows(text_test)
    text_normed_train = _normalize_rows(text_train)
    if args.alpha == 1.0:
        fusion_similarity = gnn_similarity
    elif args.alpha == 0.0:
        fusion_similarity = text_similarity
    else:
        fused_test = np.concatenate(
            [args.alpha * gnn_normed_test, (1.0 - args.alpha) * text_normed_test],
            axis=1,
        )
        fused_train = np.concatenate(
            [args.alpha * gnn_normed_train, (1.0 - args.alpha) * text_normed_train],
            axis=1,
        )
        fusion_similarity = _cosine_similarity(fused_test, fused_train)

    results = {
        "alpha": args.alpha,
        "text_only": _metrics_for_similarity(text_similarity, relevance),
        "gnn_only": _metrics_for_similarity(gnn_similarity, relevance),
        "late_fusion": _metrics_for_similarity(fusion_similarity, relevance),
    }
    logger.info("Text-only results: %s", _format_metrics("text_only", results["text_only"]))
    logger.info("GNN-only results: %s", _format_metrics("gnn_only", results["gnn_only"]))
    logger.info(
        "Late-fusion results: %s",
        _format_metrics("late_fusion", results["late_fusion"]),
    )

    output_path = Path(
        f"results/exp03_sn_text_retrieval/retrieval_results_alpha_{args.alpha}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
