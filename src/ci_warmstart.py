"""Inference-time CI embedding warmstarts from incident centroids."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData


def compute_ci_centroids(
    incident_embeddings: np.ndarray,
    data: HeteroData,
    ci_col_map: object = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average embeddings of incidents connected to each CI node.

    Non-finite incident rows are ignored, allowing callers to mask validation
    and test incidents while computing training-only centroids.
    """
    del ci_col_map
    if incident_embeddings.ndim != 2:
        raise ValueError("incident_embeddings must be two-dimensional")
    num_cis = data["ci"].num_nodes
    centroids = np.zeros((num_cis, incident_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(num_cis, dtype=np.int64)
    sums = np.zeros_like(centroids)
    edge_items = list(data.edge_index_dict.items())
    forward_edges = [
        (edge_type, edge_index)
        for edge_type, edge_index in edge_items
        if edge_type[0] == "incident" and edge_type[2] == "ci"
    ]
    selected_edges = forward_edges or [
        (edge_type, edge_index)
        for edge_type, edge_index in edge_items
        if edge_type[0] == "ci" and edge_type[2] == "incident"
    ]
    for edge_type, edge_index in selected_edges:
        if edge_type[0] == "incident":
            incident_indices = edge_index[0].cpu().numpy()
            ci_indices = edge_index[1].cpu().numpy()
        else:
            ci_indices = edge_index[0].cpu().numpy()
            incident_indices = edge_index[1].cpu().numpy()
        valid = np.isfinite(incident_embeddings[incident_indices]).all(axis=1)
        for incident_index, ci_index in zip(
            incident_indices[valid], ci_indices[valid]
        ):
            sums[ci_index] += incident_embeddings[incident_index]
            counts[ci_index] += 1
    nonzero = counts > 0
    centroids[nonzero] = sums[nonzero] / counts[nonzero, None]
    return centroids, counts


def build_hybrid_ci_embeddings(
    model: torch.nn.Module,
    data: HeteroData,
    ci_centroids: np.ndarray,
    train_ci_indices: set[int] | np.ndarray,
    device: str = "cpu",
) -> tuple[torch.Tensor, int, int, int]:
    """Combine learned embeddings for seen CIs with centroid embeddings."""
    del data
    if ci_centroids.ndim != 2:
        raise ValueError("ci_centroids must be two-dimensional")
    learned = model.entity_embeddings["ci"].weight.detach().cpu()
    if learned.shape != torch.Size(ci_centroids.shape):
        raise ValueError("ci_centroids shape must match learned CI embeddings")
    train_indices = set(np.asarray(list(train_ci_indices), dtype=np.int64).tolist())
    hybrid = torch.as_tensor(ci_centroids, dtype=torch.float32).clone()
    for ci_index in train_indices:
        if 0 <= ci_index < len(hybrid):
            hybrid[ci_index] = learned[ci_index]
    unseen_mask = np.ones(len(hybrid), dtype=bool)
    if train_indices:
        unseen_mask[list(train_indices)] = False
    num_unseen = int(unseen_mask.sum())
    num_zero = int(np.all(ci_centroids[unseen_mask] == 0, axis=1).sum())
    return hybrid.to(torch.device(device)), len(train_indices), num_unseen, num_zero


def compute_structural_ci_centroids(
    model: torch.nn.Module,
    data: HeteroData,
    train_ci_indices: set[int] | np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, int]]:
    """Build unseen CI embeddings from subtype, category, or global neighbors."""
    del device
    learned = model.entity_embeddings["ci"].weight.detach().cpu().numpy()
    num_cis, embedding_dim = learned.shape
    seen = set(np.asarray(list(train_ci_indices), dtype=np.int64).tolist())
    seen &= set(range(num_cis))
    unseen = set(range(num_cis)) - seen
    hybrid = learned.copy()
    global_mean = learned[list(seen)].mean(axis=0) if seen else np.zeros(embedding_dim)

    subtype_to_cis: dict[int, set[int]] = {}
    ci_to_subtypes: dict[int, set[int]] = {}
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_type[0] == "ci" and edge_type[2] == "ci_subtype":
            ci_nodes = edge_index[0].cpu().numpy()
            subtype_nodes = edge_index[1].cpu().numpy()
        elif edge_type[0] == "ci_subtype" and edge_type[2] == "ci":
            subtype_nodes = edge_index[0].cpu().numpy()
            ci_nodes = edge_index[1].cpu().numpy()
        else:
            continue
        for ci_index, subtype_index in zip(ci_nodes, subtype_nodes):
            subtype_to_cis.setdefault(int(subtype_index), set()).add(int(ci_index))
            ci_to_subtypes.setdefault(int(ci_index), set()).add(int(subtype_index))

    incident_to_cis: dict[int, set[int]] = {}
    incident_to_categories: dict[int, set[int]] = {}
    category_to_cis: dict[int, set[int]] = {}
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_type[0] == "incident" and edge_type[2] == "ci":
            incidents, cis = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
            for incident, ci in zip(incidents, cis):
                incident_to_cis.setdefault(int(incident), set()).add(int(ci))
        elif edge_type[0] == "incident" and edge_type[2] == "category":
            incidents, categories = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
            for incident, category in zip(incidents, categories):
                incident_to_categories.setdefault(int(incident), set()).add(int(category))

    for incident, categories in incident_to_categories.items():
        for category in categories:
            category_to_cis.setdefault(category, set()).update(incident_to_cis.get(incident, set()))

    stats = {
        "num_seen": len(seen),
        "num_unseen": len(unseen),
        "num_subtype_match": 0,
        "num_category_fallback": 0,
        "num_global_fallback": 0,
    }
    for ci_index in unseen:
        candidates = set().union(
            *(subtype_to_cis[subtype] for subtype in ci_to_subtypes.get(ci_index, set()))
        ) & seen
        if candidates:
            hybrid[ci_index] = learned[list(candidates)].mean(axis=0)
            stats["num_subtype_match"] += 1
            continue
        categories = set().union(
            *(incident_to_categories.get(incident, set())
              for incident, cis in incident_to_cis.items() if ci_index in cis)
        )
        candidates = set().union(*(category_to_cis[category] for category in categories)) & seen
        if candidates:
            hybrid[ci_index] = learned[list(candidates)].mean(axis=0)
            stats["num_category_fallback"] += 1
        else:
            hybrid[ci_index] = global_mean
            stats["num_global_fallback"] += 1
    return hybrid.astype(np.float32), stats


def compute_knn_ci_centroids(
    model: torch.nn.Module,
    data: HeteroData,
    train_ci_indices: set[int] | np.ndarray,
    ci_features: np.ndarray,
    ci_to_idx: dict[str, int],
    graph_ci_names: list[str],
    k: int = 5,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build unseen CI embeddings from k-nearest seen CI feature vectors."""
    del data, device
    if k < 1:
        raise ValueError("k must be positive")
    if ci_features.ndim != 2:
        raise ValueError("ci_features must be two-dimensional")
    learned = model.entity_embeddings["ci"].weight.detach().cpu().numpy()
    if len(graph_ci_names) != len(learned):
        raise ValueError("graph_ci_names must match learned CI embedding rows")
    aligned = np.zeros((len(graph_ci_names), ci_features.shape[1]), dtype=np.float32)
    for graph_index, ci_name in enumerate(graph_ci_names):
        feature_index = ci_to_idx.get(str(ci_name))
        if feature_index is not None:
            aligned[graph_index] = ci_features[feature_index]

    valid_indices = set(np.asarray(list(train_ci_indices), dtype=np.int64).tolist())
    valid_indices &= set(range(len(learned)))
    seen = np.array(sorted(valid_indices), dtype=np.int64)
    unseen = np.array(sorted(set(range(len(learned))) - valid_indices), dtype=np.int64)
    hybrid = learned.copy()
    global_mean = learned[seen].mean(axis=0) if len(seen) else np.zeros(learned.shape[1])
    seen_features = aligned[seen]
    seen_norms = np.linalg.norm(seen_features, axis=1)
    normalized_seen = np.divide(
        seen_features,
        seen_norms[:, None],
        out=np.zeros_like(seen_features),
        where=seen_norms[:, None] != 0,
    )
    similarities: list[float] = []
    knn_count = 0
    global_count = 0
    neighbor_count = min(k, len(seen))
    for ci_index in unseen:
        feature = aligned[ci_index]
        norm = np.linalg.norm(feature)
        if norm == 0 or neighbor_count == 0:
            hybrid[ci_index] = global_mean
            global_count += 1
            continue
        scores = normalized_seen @ (feature / norm)
        top_positions = np.argsort(scores)[-neighbor_count:]
        hybrid[ci_index] = learned[seen[top_positions]].mean(axis=0)
        similarities.append(float(scores[top_positions].mean()))
        knn_count += 1
    stats: dict[str, float | int] = {
        "num_seen": len(seen),
        "num_unseen": len(unseen),
        "num_knn_match": knn_count,
        "num_global_fallback": global_count,
        "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
    }
    return hybrid.astype(np.float32), stats
