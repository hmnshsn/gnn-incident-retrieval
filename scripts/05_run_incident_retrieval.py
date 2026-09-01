"""Run exp03 graded incident-to-incident retrieval."""

from __future__ import annotations

import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.baselines_retrieval import (
    category_similarity,
    ci_match_similarity,
    ci_subtype_similarity,
    random_similarity,
)
from src.data_loader import load_bpi2014_incidents, temporal_split
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.graph_builder import build_incident_graph_no_target
from src.logging_config import setup_logging
from src.models import HeteroIncidentClassifier, train_minibatch
from src.relevance import compute_relevance_matrix


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
RETRIEVAL_KS = (1, 3, 5, 10, 20)


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


def _evaluate_similarity(
    name: str,
    similarities: np.ndarray,
    relevance: object,
) -> tuple[str, dict[str, float]]:
    """Evaluate one similarity matrix and return its named result."""
    return name, evaluate_retrieval_graded(
        similarities,
        relevance,
        ks=RETRIEVAL_KS,
    )


def _capture_classifier_hidden(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    captured: list[torch.Tensor],
) -> None:
    """Capture classifier input, which is the final incident representation."""
    captured.append(inputs[0])


def _extract_incident_embeddings(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    incident_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Extract final hidden incident embeddings using sampled inference."""
    mask = torch.zeros(data["incident"].num_nodes, dtype=torch.bool)
    mask[torch.as_tensor(incident_indices, dtype=torch.long)] = True
    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", mask),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    captured: list[torch.Tensor] = []
    hook = model.classifier.register_forward_pre_hook(
        partial(_capture_classifier_hidden, captured=captured)
    )
    model.eval()
    embeddings: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                seed_count = batch["incident"].batch_size
                captured.clear()
                model(batch)
                if not captured:
                    raise RuntimeError("classifier hook did not capture embeddings")
                embeddings.append(captured[0][:seed_count].cpu())
    finally:
        hook.remove()
    if not embeddings:
        raise ValueError("incident_indices selects no incidents")
    return torch.cat(embeddings)


def _cosine_similarity(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute batched cosine similarities and return a dense NumPy matrix."""
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
    candidate_embeddings = candidate_embeddings.to(device)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for query_start in range(0, len(query_embeddings), 512):
            query_chunk = query_embeddings[query_start : query_start + 512].to(device)
            chunks.append((query_chunk @ candidate_embeddings.t()).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _format_table(results: dict[str, dict[str, float]]) -> str:
    """Format retrieval metrics as a fixed-width comparison table."""
    header = "Method              nDCG@1  nDCG@3  nDCG@5  nDCG@10 nDCG@20 MAP     MRR"
    lines = [header]
    for method, metrics in results.items():
        lines.append(
            f"{method:<19}"
            f"{metrics['nDCG@1']:.3f}   "
            f"{metrics['nDCG@3']:.3f}   "
            f"{metrics['nDCG@5']:.3f}   "
            f"{metrics['nDCG@10']:.3f}   "
            f"{metrics['nDCG@20']:.3f}   "
            f"{metrics['MAP']:.3f}   "
            f"{metrics['MRR']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run baselines and GNN graded retrieval, then save all metrics."""
    logger = setup_logging("bpi_exp03_retrieval")
    _set_seed(42)
    df = load_bpi2014_incidents()
    train_df, val_df, test_df = temporal_split(df)
    train_indices, val_indices, test_indices = _split_positions(len(df))
    if (
        len(train_df) != len(train_indices)
        or len(val_df) != len(val_indices)
        or len(test_df) != len(test_indices)
    ):
        raise RuntimeError("temporal split lengths do not match positional indices")

    relevance = compute_relevance_matrix(df, train_indices, test_indices)
    results: dict[str, dict[str, float]] = {}
    baseline_functions = (
        ("Random", random_similarity(len(test_indices), len(train_indices))),
        ("CI-Match", ci_match_similarity(df, train_indices, test_indices)),
        ("CI-Subtype", ci_subtype_similarity(df, train_indices, test_indices)),
        ("Category", category_similarity(df, train_indices, test_indices)),
    )
    for name, similarities in baseline_functions:
        result_name, metrics = _evaluate_similarity(name, similarities, relevance)
        results[result_name] = metrics
        logger.info("Baseline evaluation: method=%s, metrics=%s", result_name, metrics)

    data = build_incident_graph_no_target(df)
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
        num_classes=int(df["Closure Code"].nunique()),
        num_layers=2,
        dropout=0.3,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Training started")
    training_start = time.perf_counter()
    model, history = train_minibatch(
        model,
        data,
        torch.as_tensor(train_indices, dtype=torch.long),
        torch.as_tensor(val_indices, dtype=torch.long),
        epochs=50,
        lr=0.005,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        device=str(device),
    )
    logger.info("Training completed: duration_seconds=%.2f", time.perf_counter() - training_start)
    train_embeddings = _extract_incident_embeddings(
        model, data, train_indices, device
    )
    test_embeddings = _extract_incident_embeddings(
        model, data, test_indices, device
    )
    gnn_similarity = _cosine_similarity(test_embeddings, train_embeddings, device)
    _, results["GNN"] = _evaluate_similarity("GNN", gnn_similarity, relevance)
    logger.info("GNN evaluation: metrics=%s", results["GNN"])

    results_table = _format_table(results)
    logger.info("Final results table:\n%s", results_table)
    output_path = Path("results/exp03_incident_retrieval/retrieval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metrics": results,
                "history": history,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "relevance_shape": list(relevance.shape),
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
