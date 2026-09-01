"""Compare fixed and CI-visibility-adaptive SN retrieval fusion."""

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
METRIC_KEYS = ("ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20", "map", "mrr")


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


def _fused_similarity(
    query_gnn: np.ndarray,
    train_gnn: np.ndarray,
    query_text: np.ndarray,
    train_text: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute late-fusion cosine similarity for one alpha value."""
    query_fused = np.concatenate(
        [alpha * _normalize_rows(query_gnn), (1.0 - alpha) * _normalize_rows(query_text)],
        axis=1,
    )
    train_fused = np.concatenate(
        [alpha * _normalize_rows(train_gnn), (1.0 - alpha) * _normalize_rows(train_text)],
        axis=1,
    )
    return _cosine_similarity(query_fused, train_fused)


def _evaluate_group(
    gnn_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    dataframe: object,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Evaluate one test group against the full training candidate set."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    similarities = _fused_similarity(
        gnn_embeddings[test_indices],
        gnn_embeddings[train_indices],
        text_embeddings[test_indices],
        text_embeddings[train_indices],
        alpha,
    )
    return _metrics_for_similarity(similarities, relevance)


def _weighted_metrics(
    seen_metrics: dict[str, float],
    unseen_metrics: dict[str, float],
    seen_count: int,
    unseen_count: int,
) -> dict[str, float]:
    """Combine group-average metrics weighted by group query counts."""
    total = seen_count + unseen_count
    if total == 0:
        raise ValueError("no test incidents available")
    return {
        key: (
            seen_metrics[key] * seen_count + unseen_metrics[key] * unseen_count
        )
        / total
        for key in METRIC_KEYS
    }


def _format_table(
    fixed_results: dict[str, dict[str, float]],
    adaptive_results: dict[str, dict[str, float]],
) -> str:
    """Format fixed and adaptive metrics as a comparison table."""
    lines = ["Method       Group   nDCG@1  MAP     MRR"]
    for method, results in (("fixed_0.8", fixed_results), ("adaptive", adaptive_results)):
        for group in ("seen", "unseen", "all"):
            metrics = results[group]
            lines.append(
                f"{method:<12}{group:<8}"
                f"{metrics['ndcg@1']:.3f}   "
                f"{metrics['map']:.3f}   "
                f"{metrics['mrr']:.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    """Run adaptive-alpha cold-start analysis and save comparison metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-seen", type=float, default=0.9)
    parser.add_argument("--alpha-unseen", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, alpha in (
        ("alpha-seen", args.alpha_seen),
        ("alpha-unseen", args.alpha_unseen),
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
    logger = setup_logging("sn_exp06_adaptive_alpha")
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
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]
    seen_count = len(seen_test_indices)
    unseen_count = len(unseen_test_indices)

    adaptive_seen = _evaluate_group(
        gnn_embeddings,
        text_embeddings,
        dataframe,
        train_indices,
        seen_test_indices,
        args.alpha_seen,
    )
    adaptive_unseen = _evaluate_group(
        gnn_embeddings,
        text_embeddings,
        dataframe,
        train_indices,
        unseen_test_indices,
        args.alpha_unseen,
    )
    adaptive = {
        "seen": adaptive_seen,
        "unseen": adaptive_unseen,
        "all": _weighted_metrics(
            adaptive_seen,
            adaptive_unseen,
            seen_count,
            unseen_count,
        ),
    }

    fixed_seen = _evaluate_group(
        gnn_embeddings,
        text_embeddings,
        dataframe,
        train_indices,
        seen_test_indices,
        0.8,
    )
    fixed_unseen = _evaluate_group(
        gnn_embeddings,
        text_embeddings,
        dataframe,
        train_indices,
        unseen_test_indices,
        0.8,
    )
    fixed_results = {
        "seen": fixed_seen,
        "unseen": fixed_unseen,
        "all": _weighted_metrics(fixed_seen, fixed_unseen, seen_count, unseen_count),
    }
    logger.info("Adaptive alpha comparison:\n%s", _format_table(fixed_results, adaptive))

    output_path = Path("results/exp06_sn_adaptive_alpha/adaptive_alpha_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "alpha_seen": args.alpha_seen,
                "alpha_unseen": args.alpha_unseen,
                "num_test_seen": seen_count,
                "num_test_unseen": unseen_count,
                "adaptive": adaptive,
                "fixed_0.8": fixed_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
