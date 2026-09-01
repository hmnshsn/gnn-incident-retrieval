"""Evaluation metrics for graded incident retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse


def dcg_at_k(relevances: Sequence[float] | np.ndarray, k: int) -> float:
    """Compute discounted cumulative gain at rank ``k``.

    Graded gains use ``2**relevance - 1`` and rank denominators use
    ``log2(i + 1)`` for one-indexed rank ``i``.

    Args:
        relevances: Relevance values in ranked order.
        k: Maximum rank included in the calculation.

    Returns:
        DCG value as a float.
    """
    if k < 1:
        return 0.0
    values = np.asarray(relevances, dtype=float)[:k]
    if values.size == 0:
        return 0.0
    ranks = np.arange(2, values.size + 2, dtype=float)
    return float(np.sum(np.expm1(np.log(2.0) * values) / np.log2(ranks)))


def ndcg_at_k(relevances: Sequence[float] | np.ndarray, k: int) -> float:
    """Compute normalized discounted cumulative gain at rank ``k``.

    Args:
        relevances: Relevance values in predicted ranked order.
        k: Maximum rank included in the calculation.

    Returns:
        nDCG value in ``[0, 1]``; returns 0 when ideal DCG is zero.
    """
    values = np.asarray(relevances, dtype=float)
    dcg = dcg_at_k(values, k)
    ideal = dcg_at_k(np.sort(values)[::-1], k)
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retrieval_graded(
    similarity_matrix: np.ndarray | sparse.spmatrix,
    relevance_matrix: sparse.spmatrix,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Evaluate ranked retrieval with nDCG, MAP, and MRR.

    Similarities are sorted in descending order independently for each query.
    Any positive relevance counts as relevant for MAP and MRR; nDCG preserves
    the full graded values.

    Args:
        similarity_matrix: Dense or sparse array of shape ``(n_queries,
            n_candidates)``.
        relevance_matrix: Sparse relevance matrix with the same shape.
        ks: Cutoffs for nDCG metrics.

    Returns:
        Dictionary containing ``nDCG@k`` for each cutoff, ``MAP``, and ``MRR``.
    """
    similarities = (
        similarity_matrix.toarray()
        if sparse.issparse(similarity_matrix)
        else np.asarray(similarity_matrix)
    )
    relevance = relevance_matrix.tocsr()
    if similarities.ndim != 2 or relevance.shape != similarities.shape:
        raise ValueError("similarity and relevance matrices must have matching 2-D shapes")
    if any(k < 1 for k in ks):
        raise ValueError("all ks must be positive")

    ndcg_totals = {f"nDCG@{k}": 0.0 for k in ks}
    average_precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for query_index in range(similarities.shape[0]):
        order = np.argsort(-similarities[query_index], kind="stable")
        ranked_relevance = relevance.getrow(query_index).toarray().ravel()[order]
        for k in ks:
            ndcg_totals[f"nDCG@{k}"] += ndcg_at_k(ranked_relevance, k)

        relevant = ranked_relevance > 0
        relevant_count = int(relevant.sum())
        if relevant_count == 0:
            average_precisions.append(0.0)
            reciprocal_ranks.append(0.0)
            continue
        cumulative_relevant = np.cumsum(relevant)
        ranks = np.arange(1, len(relevant) + 1)
        precision_at_rank = cumulative_relevant / ranks
        average_precisions.append(float(precision_at_rank[relevant].sum() / relevant_count))
        first_rank = int(np.flatnonzero(relevant)[0]) + 1
        reciprocal_ranks.append(1.0 / first_rank)

    query_count = similarities.shape[0]
    if query_count == 0:
        raise ValueError("similarity matrix contains no queries")
    metrics = {
        key: value / query_count for key, value in ndcg_totals.items()
    }
    metrics["MAP"] = float(np.mean(average_precisions))
    metrics["MRR"] = float(np.mean(reciprocal_ranks))
    return metrics
