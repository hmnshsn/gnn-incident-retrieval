"""Leakage-safe text kNN edges for incident graphs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch_geometric.data import HeteroData


def build_text_knn_edges(
    text_embeddings: np.ndarray,
    source_indices: Iterable[int],
    train_indices: Iterable[int],
    k: int,
    chunk_size: int = 512,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Build directed edges from each source incident to nearest train incidents."""
    embeddings = np.asarray(text_embeddings, dtype=np.float32)
    sources = np.asarray(list(source_indices), dtype=np.int64)
    candidates = np.asarray(list(train_indices), dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError("text_embeddings must be a two-dimensional array")
    if np.any(sources < 0) or np.any(sources >= len(embeddings)):
        raise IndexError("source_indices contain an invalid incident index")
    if np.any(candidates < 0) or np.any(candidates >= len(embeddings)):
        raise IndexError("train_indices contain an invalid incident index")
    if k < 1 or k > len(candidates):
        raise ValueError("k must be between 1 and the number of train candidates")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0,
    )
    candidate_embeddings = normalized[candidates]
    edge_sources: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    candidate_lookup = {int(index): position for position, index in enumerate(candidates)}
    for start in range(0, len(sources), chunk_size):
        query_indices = sources[start : start + chunk_size]
        scores = normalized[query_indices] @ candidate_embeddings.T
        if exclude_self:
            for row, source in enumerate(query_indices):
                candidate_position = candidate_lookup.get(int(source))
                if candidate_position is not None:
                    scores[row, candidate_position] = -np.inf
        positions = np.argpartition(scores, kth=-k, axis=1)[:, -k:]
        selected_scores = np.take_along_axis(scores, positions, axis=1)
        order = np.argsort(selected_scores, axis=1)[:, ::-1]
        positions = np.take_along_axis(positions, order, axis=1)
        selected_scores = np.take_along_axis(selected_scores, order, axis=1)
        edge_sources.append(np.repeat(query_indices, k))
        edge_targets.append(candidates[positions].reshape(-1))
        similarities.append(selected_scores.reshape(-1))
    if edge_sources:
        source_array = np.concatenate(edge_sources)
        target_array = np.concatenate(edge_targets)
        similarity_array = np.concatenate(similarities)
        edge_index = torch.as_tensor(
            np.stack([source_array, target_array]), dtype=torch.long
        )
        finite_similarities = similarity_array[np.isfinite(similarity_array)]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        finite_similarities = np.empty(0, dtype=np.float32)
    return edge_index, {
        "k": int(k),
        "num_sources": int(len(sources)),
        "num_edges": int(edge_index.size(1)),
        "mean_similarity": float(np.mean(finite_similarities)) if len(finite_similarities) else 0.0,
        "min_similarity": float(np.min(finite_similarities)) if len(finite_similarities) else 0.0,
    }


def add_knn_relation(
    data: HeteroData,
    edge_index: torch.Tensor,
    relation_name: str = "text_similar",
) -> HeteroData:
    """Attach symmetric incident-to-incident relation to a graph in place."""
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a torch.long tensor with shape (2, num_edges)")
    forward = ("incident", relation_name, "incident")
    reverse = ("incident", f"rev_{relation_name}", "incident")
    data[forward].edge_index = edge_index
    data[reverse].edge_index = edge_index.flip(0)
    return data
