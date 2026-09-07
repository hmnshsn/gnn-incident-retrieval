"""Measure seen-CI retrieval after replacing trained CI embeddings."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from src.logging_config import setup_logging
from src.models import HeteroIncidentClassifier, extract_incident_embeddings, train_minibatch
from src.sweep_runner import (
    BATCH_SIZE,
    NUM_NEIGHBORS,
    _build_node_counts,
    _evaluate_group,
    _training_ci_indices,
    get_shared_context,
)



def _set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _test_ci_indices(data: object, num_incidents: int) -> np.ndarray:
    """Return CI node index for each incident, or -1 when no CI edge exists."""
    indices = np.full(num_incidents, -1, dtype=np.int64)
    edge_index = data["incident", "affects", "ci"].edge_index
    incident_indices = edge_index[0].cpu().numpy()
    ci_indices = edge_index[1].cpu().numpy()
    indices[incident_indices] = ci_indices
    return indices


def _evaluate_groups(
    embeddings: np.ndarray,
    dataframe: object,
    train_indices: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Evaluate non-empty incident groups against training incidents."""
    return {
        group: _evaluate_group(embeddings, dataframe, train_indices, indices)
        for group, indices in groups.items()
        if len(indices) > 0
    }


def _aggregate(records: list[dict]) -> dict:
    """Aggregate baseline, masked, and training metrics across successful seeds."""
    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        for embedding_type in ("baseline", "masked"):
            for group, metrics in record[embedding_type].items():
                for metric, value in metrics.items():
                    values.setdefault(embedding_type, {}).setdefault(group, {}).setdefault(metric, []).append(float(value))
    aggregated = {
        embedding_type: {
            group: {
                metric: {
                    "mean": statistics.mean(metric_values),
                    "std": statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0,
                }
                for metric, metric_values in metrics.items()
            }
            for group, metrics in groups.items()
        }
        for embedding_type, groups in values.items()
    }
    scalar_names = ("best_epoch", "epochs_trained", "best_val_loss")
    aggregated["training"] = {}
    for name in scalar_names:
        scalar_values = [float(record[name]) for record in records]
        if scalar_values:
            aggregated["training"][name] = {
                "mean": statistics.mean(scalar_values),
                "std": statistics.pstdev(scalar_values) if len(scalar_values) > 1 else 0.0,
            }
    return aggregated


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for seen-CI masking diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=["categorical", "text", "concat"], default="concat")
    parser.add_argument("--entity-embed-dim", type=int, default=64)
    parser.add_argument("--ci-dropout-rate", type=float, default=0.5)
    parser.add_argument("--ci-dropout-mode", choices=["embedding", "edge", "both"], default="embedding")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 42, 123])
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    return parser


def _run_seed(args: argparse.Namespace, context: dict, seed: int) -> dict:
    """Train one model, mask selected seen CIs, and evaluate both embeddings."""
    _set_seed(seed)
    dataframe = context["dataframe"]
    data = context["graphs"][(args.features, 0, "all")]
    train_indices = context["train_indices"]
    input_dim = int(data["incident"].x.size(1))
    model = HeteroIncidentClassifier(
        data.metadata(),
        node_counts=_build_node_counts(data),
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=context["num_classes"],
        num_layers=args.num_layers,
        dropout=args.dropout,
        ci_dropout_rate=args.ci_dropout_rate,
        ci_dropout_mode=args.ci_dropout_mode,
        entity_embed_dim=args.entity_embed_dim,
    )
    model, history = train_minibatch(
        model,
        data,
        context["masks"]["train_mask"],
        context["masks"]["val_mask"],
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        device=str(context["device"]),
        patience=args.patience,
    )
    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    baseline_embeddings = extract_incident_embeddings(
        model, data, loader, str(context["device"])
    ).numpy()
    seen_ci_indices = _training_ci_indices(data, train_indices)
    masked_count = round(args.mask_fraction * len(seen_ci_indices))
    masked_ci_indices = set(
        np.random.default_rng(seed).choice(
            sorted(seen_ci_indices), size=masked_count, replace=False
        ).tolist()
    )
    masked_ci_embeddings = model.entity_embeddings["ci"].weight.detach().cpu().clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for ci_index in masked_ci_indices:
        masked_ci_embeddings[ci_index] = torch.randn(
            masked_ci_embeddings[ci_index].shape, generator=generator
        )
    masked_embeddings = extract_incident_embeddings(
        model,
        data,
        loader,
        str(context["device"]),
        masked_ci_embeddings,
        "inductive",
    ).numpy()
    test_ci_indices = _test_ci_indices(data, len(dataframe))
    seen_test_indices = context["seen_test_indices"]
    masked_test_indices = seen_test_indices[np.isin(test_ci_indices[seen_test_indices], list(masked_ci_indices))]
    intact_test_indices = seen_test_indices[~np.isin(test_ci_indices[seen_test_indices], list(masked_ci_indices))]
    groups = {
        "seen_masked": masked_test_indices,
        "seen_intact": intact_test_indices,
        "unseen": context["unseen_test_indices"],
    }
    return {
        "seed": seed,
        "best_epoch": history["best_epoch"],
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": min(history["val_loss"]),
        "num_seen_cis": len(seen_ci_indices),
        "num_masked_cis": len(masked_ci_indices),
        "group_sizes": {group: len(indices) for group, indices in groups.items()},
        "baseline": _evaluate_groups(baseline_embeddings, dataframe, train_indices, groups),
        "masked": _evaluate_groups(masked_embeddings, dataframe, train_indices, groups),
    }


def main() -> None:
    """Run seen-CI masking diagnostic across requested seeds."""
    args = _build_parser().parse_args()
    if not 0.0 <= args.mask_fraction <= 1.0:
        raise ValueError("mask-fraction must be between 0.0 and 1.0")
    logger = setup_logging("seen_ci_masking")
    context = get_shared_context(
        args.subtype,
        [args.features],
        knn_edges=0,
        knn_scope="all",
    )
    records: list[dict] = []
    failures: list[dict] = []
    for seed in args.seeds:
        try:
            records.append(_run_seed(args, context, seed))
        except Exception as error:
            logger.exception("Seed %d failed", seed)
            failures.append({"seed": seed, "error": str(error)})
    aggregated = _aggregate(records)
    logger.info("Group            baseline nDCG@1/MAP    masked nDCG@1/MAP")
    for group in ("seen_masked", "seen_intact", "unseen"):
        baseline = aggregated.get("baseline", {}).get(group, {})
        masked = aggregated.get("masked", {}).get(group, {})
        if not baseline and not masked:
            continue
        logger.info(
            "%-16s %.3f ± %.3f / %.3f ± %.3f    %.3f ± %.3f / %.3f ± %.3f",
            group,
            baseline.get("ndcg@1", {}).get("mean", 0.0), baseline.get("ndcg@1", {}).get("std", 0.0),
            baseline.get("map", {}).get("mean", 0.0), baseline.get("map", {}).get("std", 0.0),
            masked.get("ndcg@1", {}).get("mean", 0.0), masked.get("ndcg@1", {}).get("std", 0.0),
            masked.get("map", {}).get("mean", 0.0), masked.get("map", {}).get("std", 0.0),
        )
    output = {
        "config": vars(args),
        "records": records,
        "aggregated": aggregated,
        "failures": failures,
    }
    output_path = Path("results/diagnostics/seen_ci_masking.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
