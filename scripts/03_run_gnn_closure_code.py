"""Run the target-free minibatch GNN experiment."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data_loader import load_bpi2014_incidents
from src.evaluate import evaluate_retrieval
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.logging_config import setup_logging
from src.models import HeteroIncidentClassifier, predict_ranks, train_minibatch


def _set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    """Format test metrics as a fixed-width comparison table."""
    lines = ["Method              MRR     Hits@1  Hits@3  Hits@5  Hits@10"]
    for method, metrics in rows:
        lines.append(
            f"{method:<19}"
            f"{metrics['MRR']:.3f}   "
            f"{metrics['Hits@1']:.3f}   "
            f"{metrics['Hits@3']:.3f}   "
            f"{metrics['Hits@5']:.3f}   "
            f"{metrics['Hits@10']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Build, train, evaluate, and save the v2 GNN experiment."""
    logger = setup_logging("bpi_exp02_closure_code")
    _set_seed(42)
    df = load_bpi2014_incidents()
    masks = make_split_masks(df)
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
    logger.info("Training started")
    training_start = time.perf_counter()
    model, history = train_minibatch(
        model,
        data,
        masks["train_mask"],
        masks["val_mask"],
        epochs=50,
        lr=0.005,
        batch_size=1024,
        device="cuda:0",
    )
    logger.info("Training completed: duration_seconds=%.2f", time.perf_counter() - training_start)

    split_metrics: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        ranks = predict_ranks(
            model,
            data,
            masks[f"{split_name}_mask"],
            device="cuda:0",
            batch_size=1024,
        )
        split_metrics[split_name] = evaluate_retrieval(ranks)

    rows = [("GNN", split_metrics["test"])]
    baseline_path = Path("results/baselines.json")
    baseline_results = {}
    if baseline_path.exists():
        baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
        rows.extend(baseline_results.items())
    for method, metrics in baseline_results.items():
        logger.info("Baseline evaluation: method=%s, metrics=%s", method, metrics)
    logger.info("GNN evaluation: metrics=%s", split_metrics["test"])
    results_table = _format_table(rows)
    logger.info("Final results table:\n%s", results_table)

    output_path = Path("results/gnn_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metrics": split_metrics,
                "history": history,
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
