"""Retrieval evaluation: rank computation and Hits@k / MRR metrics.

Ranking is performed with cosine similarity between query and candidate
embeddings. Ranks are 1-indexed (rank 1 == top match).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def compute_ranks(
    query_embs: torch.Tensor,
    candidate_embs: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """Compute the rank of the ground-truth candidate for each query.

    For each query the cosine similarity against every candidate is computed,
    candidates are sorted by descending similarity, and the 1-indexed rank of
    the ground-truth candidate is returned.

    Args:
        query_embs: Tensor of shape ``(Q, D)`` with query embeddings.
        candidate_embs: Tensor of shape ``(C, D)`` with candidate embeddings.
            When ``C == Q`` (one candidate set per query) the same candidate
            pool is used for all queries; otherwise ``candidate_embs`` must be
            shape ``(Q, D)`` and is treated as a per-query candidate pool of
            size 1 (not supported here).
        ground_truth: Tensor of shape ``(Q,)`` with the integer candidate index
            that is the correct match for each query.

    Returns:
        LongTensor of shape ``(Q,)`` with 1-indexed ranks (1 == best).
    """
    if query_embs.dim() != 2 or candidate_embs.dim() != 2:
        raise ValueError("query_embs and candidate_embs must be 2-D")
    if query_embs.size(-1) != candidate_embs.size(-1):
        raise ValueError("query and candidate embedding dims must match")
    if ground_truth.dim() != 1 or ground_truth.size(0) != query_embs.size(0):
        raise ValueError("ground_truth must be 1-D with one entry per query")

    query_embs = F.normalize(query_embs, p=2, dim=-1)
    candidate_embs = F.normalize(candidate_embs, p=2, dim=-1)

    # Similarity matrix (Q, C).
    sim = query_embs @ candidate_embs.t()

    # Sort candidates by descending similarity for each query.
    # argsort gives indices of ascending order; flip to get descending.
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    # For each query, find the position of its ground-truth candidate.
    ranks = torch.empty(query_embs.size(0), dtype=torch.long, device=query_embs.device)
    for i in range(query_embs.size(0)):
        pos = (sorted_idx[i] == ground_truth[i]).nonzero(as_tuple=True)[0]
        ranks[i] = pos.item() + 1  # 1-indexed
    return ranks


def evaluate_retrieval(
    ranks: torch.Tensor,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Summarise retrieval quality from a tensor of 1-indexed ranks.

    Args:
        ranks: 1-D LongTensor of 1-indexed ranks (1 == best) for each query.
        ks: Values of ``k`` for which to compute Hits@k.

    Returns:
        Dict with ``MRR`` plus ``Hits@{k}`` for each requested ``k``. Values
        are floats in ``[0, 1]``.
    """
    if ranks.dim() != 1:
        raise ValueError("ranks must be a 1-D tensor")
    ranks = ranks.to(torch.long)
    n = ranks.numel()
    if n == 0:
        raise ValueError("ranks tensor is empty")

    ranks_f = ranks.to(torch.float32)
    mrr = float((1.0 / ranks_f).mean().item())

    metrics: dict[str, float] = {"MRR": mrr}
    for k in ks:
        hits = float((ranks <= k).to(torch.float32).mean().item())
        metrics[f"Hits@{k}"] = hits
    return metrics
