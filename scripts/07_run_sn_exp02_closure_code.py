"""Run exp02 closure-code prediction on ServiceNow incidents."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from src.baselines import (
    category_match_baseline,
    ci_majority_baseline,
    random_baseline,
    text_similarity_baseline,
)
from src.data_loader_sn import load_sn_incidents
from src.evaluate import evaluate_retrieval
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.models import HeteroIncidentClassifier, predict_ranks, train_minibatch



def _set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_table(results: dict[str, dict[str, float]]) -> str:
    """Format test metrics as a fixed-width comparison table."""
    lines = ["Method              MRR     Hits@1  Hits@3  Hits@5  Hits@10"]
    for method, metrics in results.items():
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
    """Run SN closure-code baselines and GNN evaluation."""
    _set_seed(42)
    df = load_sn_incidents()
    df = df.dropna(subset=["Closure Code"]).reset_index(drop=True)
    masks = make_split_masks(df)
    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]
    test_mask = masks["test_mask"]
    train_indices = torch.where(train_mask)[0].numpy()
    val_indices = torch.where(val_mask)[0].numpy()
    test_indices = torch.where(test_mask)[0].numpy()
    split_indices = (train_indices, val_indices, test_indices)

    n_candidates = df["Closure Code"].nunique()
    baseline_results = {
        "Random": evaluate_retrieval(
            random_baseline(len(test_indices), n_candidates, seed=42)
        ),
        "Category-Match": evaluate_retrieval(
            category_match_baseline(df, split_indices)
        ),
        "CI-Majority": evaluate_retrieval(ci_majority_baseline(df, split_indices)),
        "Text-Similarity": evaluate_retrieval(
            text_similarity_baseline(df, split_indices)
        ),
    }

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
        num_classes=n_candidates,
        num_layers=2,
        dropout=0.3,
    )
    model, history = train_minibatch(
        model,
        data,
        train_mask,
        val_mask,
        epochs=50,
        lr=0.005,
        batch_size=1024,
        device="cuda:0",
    )
    gnn_ranks = predict_ranks(model, data, test_mask, device="cuda:0")
    results = {**baseline_results, "GNN": evaluate_retrieval(gnn_ranks)}
    print(_format_table(results))

    output_path = Path("results/exp02_sn_closure_code/gnn_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metrics": results["GNN"],
                "baseline_metrics": baseline_results,
                "history": history,
                "num_classes": n_candidates,
                "n_incidents": len(df),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
