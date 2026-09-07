"""Run sequential wandb regularization sweeps for SN incident retrieval."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()
os.environ["WANDB__REQUIRE_CORE"] = "false"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"

import wandb
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.data_loader import temporal_split
from src.data_loader_sn import load_sn_incidents
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.impute_subcategory import impute_subcategory
from src.logging_config import setup_logging
from src.models import (
    HeteroIncidentClassifier,
    extract_incident_embeddings,
    train_minibatch,
)
from src.relevance import compute_relevance_matrix


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
SEED = 42
WANDB_PROJECT = "gnn-incident-retrieval"
ENTITY_EMBED_DIMS = [32, 64, 128]
CI_DROPOUT_RATES = [0.1, 0.2, 0.3, 0.5]
LEARNING_RATES = [5e-4, 1e-3, 2e-3]


def _set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_dataframe(
    subtype_mode: str,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Load SN data, apply subtype mode, and track graph-row positions."""
    dataframe = load_sn_incidents()
    original_indices = np.arange(len(dataframe), dtype=np.int64)
    if subtype_mode == "imputed":
        dataframe = impute_subcategory(dataframe, int(len(dataframe) * 0.7))
    elif subtype_mode == "none":
        dataframe["CI Subtype (aff)"] = np.nan
    valid_rows = dataframe["Closure Code"].notna()
    graph_row_indices = original_indices[valid_rows.to_numpy()]
    dataframe = dataframe.loc[valid_rows].reset_index(drop=True)
    return dataframe, temporal_split(dataframe, time_col="opened_at"), graph_row_indices


def _split_positions(
    n_rows: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positional train, validation, and test indices."""
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    return (
        np.arange(train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n_rows),
    )


def _evaluate_group(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    """Evaluate one test CI-visibility group against all train candidates."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    query = embeddings[test_indices]
    candidates = embeddings[train_indices]
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    similarities = np.divide(
        query,
        query_norm,
        out=np.zeros_like(query, dtype=np.float32),
        where=query_norm != 0,
    ) @ np.divide(
        candidates,
        candidate_norm,
        out=np.zeros_like(candidates, dtype=np.float32),
        where=candidate_norm != 0,
    ).T
    metrics = evaluate_retrieval_graded(similarities, relevance, ks=(1, 5, 10, 20))
    return {
        "ndcg@1": metrics["nDCG@1"],
        "ndcg@5": metrics["nDCG@5"],
        "map": metrics["MAP"],
        "mrr": metrics["MRR"],
    }


def _training_ci_indices(data: HeteroData, train_indices: np.ndarray) -> set[int]:
    """Find CI nodes connected to training incidents."""
    train_mask = np.zeros(data["incident"].num_nodes, dtype=bool)
    train_mask[train_indices] = True
    seen: set[int] = set()
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_type[0] == "incident" and edge_type[2] == "ci":
            incident_indices, ci_indices = (
                edge_index[0].cpu().numpy(),
                edge_index[1].cpu().numpy(),
            )
        elif edge_type[0] == "ci" and edge_type[2] == "incident":
            ci_indices, incident_indices = (
                edge_index[0].cpu().numpy(),
                edge_index[1].cpu().numpy(),
            )
        else:
            continue
        seen.update(ci_indices[train_mask[incident_indices]].tolist())
    return seen


def _build_node_counts(data: HeteroData) -> dict[str, int]:
    """Extract positive non-incident node counts from a graph."""
    return {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }


def _build_graph(
    dataframe: pd.DataFrame,
    features: str,
    text_embeddings: np.ndarray | None,
) -> HeteroData:
    """Build one fresh graph for a sweep configuration."""
    if features == "text":
        return build_incident_graph_no_target(
            dataframe,
            incident_features=text_embeddings,
        )
    data = build_incident_graph_no_target(dataframe)
    if features == "concat":
        if text_embeddings is None:
            raise ValueError("text embeddings required for concat features")
        base_features = data["incident"].x.numpy()
        if base_features.shape[0] != text_embeddings.shape[0]:
            raise ValueError("base and text feature row counts do not match")
        data["incident"].x = torch.as_tensor(
            np.concatenate([base_features, text_embeddings], axis=1),
            dtype=torch.float32,
        )
    return data


def _run_name(entity_embed_dim: int, ci_dropout_rate: float, lr: float) -> str:
    """Build stable wandb and local output name."""
    return f"embed{entity_embed_dim}_drop{ci_dropout_rate}_lr{lr}"


def main() -> None:
    """Run regularization grid sequentially and log each configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-embed-dims", nargs="+", type=int, default=ENTITY_EMBED_DIMS)
    parser.add_argument("--ci-dropout-rates", nargs="+", type=float, default=CI_DROPOUT_RATES)
    parser.add_argument("--lrs", nargs="+", type=float, default=LEARNING_RATES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ci-dropout-mode", choices=["embedding", "edge", "both"], default="embedding")
    parser.add_argument("--features", choices=["tfidf", "text", "concat"], default="tfidf")
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    args = parser.parse_args()

    grid = list(itertools.product(
        args.entity_embed_dims,
        args.ci_dropout_rates,
        args.lrs,
    ))
    if args.dry_run:
        for entity_embed_dim, ci_dropout_rate, lr in grid:
            print(_run_name(entity_embed_dim, ci_dropout_rate, lr))
        print(f"Total runs: {len(grid)}")
        return

    logger = setup_logging("sn_exp13_regularization_sweep")
    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY is not set")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if any(value < 1 for value in args.entity_embed_dims):
        raise ValueError("entity embedding dimensions must be positive")
    if any(value < 0 or value > 1 for value in args.ci_dropout_rates):
        raise ValueError("CI dropout rates must be between 0 and 1")
    if any(value <= 0 for value in args.lrs):
        raise ValueError("learning rates must be positive")

    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    masks = make_split_masks(dataframe)
    text_embeddings = None
    if args.features in {"text", "concat"}:
        text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/exp13a_regularization_sweep_broad")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(grid)
    completed_runs: list[dict[str, object]] = []

    for run_idx, (entity_embed_dim, ci_dropout_rate, lr) in enumerate(grid):
        run = None
        run_name = _run_name(entity_embed_dim, ci_dropout_rate, lr)
        run_started = time.perf_counter()
        try:
            _set_seed(args.seed)
            data = _build_graph(dataframe, args.features, text_embeddings)
            node_counts = _build_node_counts(data)
            model = HeteroIncidentClassifier(
                data.metadata(),
                node_counts=node_counts,
                input_dim=data["incident"].x.size(1),
                hidden_dim=128,
                num_classes=int(dataframe["Closure Code"].nunique()),
                num_layers=2,
                dropout=0.3,
                ci_dropout_rate=ci_dropout_rate,
                ci_dropout_mode=args.ci_dropout_mode,
                entity_embed_dim=entity_embed_dim,
            )
            config = {
                "entity_embed_dim": entity_embed_dim,
                "ci_dropout_rate": ci_dropout_rate,
                "ci_dropout_mode": args.ci_dropout_mode,
                "lr": lr,
                "weight_decay": args.weight_decay,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "features": args.features,
                "subtype": args.subtype,
                "epochs_max": args.epochs,
                "patience": args.patience,
                "batch_size": BATCH_SIZE,
                "num_neighbors": NUM_NEIGHBORS,
                "seed": args.seed,
                "num_cis": data["ci"].num_nodes,
                "num_incidents": len(dataframe),
                "num_train": len(train_indices),
                "num_test": len(test_indices),
                "num_test_seen": len(seen_test_indices),
                "num_test_unseen": len(unseen_test_indices),
            }
            run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                config=config,
                tags=["regularization_sweep"] + args.wandb_tags,
                reinit="finish_previous",
            )
            model, history = train_minibatch(
                model,
                data,
                masks["train_mask"],
                masks["val_mask"],
                epochs=args.epochs,
                lr=lr,
                batch_size=BATCH_SIZE,
                num_neighbors=NUM_NEIGHBORS,
                device=str(device),
                patience=args.patience,
                weight_decay=args.weight_decay,
            )
            for epoch_idx in range(len(history["train_loss"])):
                wandb.log({
                    "epoch": epoch_idx + 1,
                    "train_loss": history["train_loss"][epoch_idx],
                    "val_loss": history["val_loss"][epoch_idx],
                    "val_acc": history["val_acc"][epoch_idx],
                })
            wandb.log({
                "best_epoch": history["best_epoch"],
                "stopped_early": history["stopped_early"],
                "epochs_trained": len(history["train_loss"]),
                "best_val_loss": min(history["val_loss"]),
            })
            full_loader = NeighborLoader(
                data,
                num_neighbors=NUM_NEIGHBORS,
                input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            embeddings = extract_incident_embeddings(
                model, data, full_loader, str(device)
            ).numpy()
            results: dict[str, dict[str, float]] = {}
            for group_name, group_indices in (
                ("seen", seen_test_indices),
                ("unseen", unseen_test_indices),
                ("all", test_indices),
            ):
                if len(group_indices) == 0:
                    continue
                metrics = _evaluate_group(
                    embeddings, dataframe, train_indices, group_indices
                )
                results[group_name] = metrics
                for metric_name, metric_value in metrics.items():
                    wandb.log({f"{group_name}/{metric_name}": metric_value})
            wandb.summary["seen_ndcg1"] = results.get("seen", {}).get("ndcg@1", 0)
            wandb.summary["unseen_ndcg1"] = results.get("unseen", {}).get("ndcg@1", 0)
            wandb.summary["all_ndcg1"] = results.get("all", {}).get("ndcg@1", 0)
            wandb.summary["unseen_map"] = results.get("unseen", {}).get("map", 0)
            wandb.summary["unseen_mrr"] = results.get("unseen", {}).get("mrr", 0)
            wandb.summary["seen_unseen_gap"] = (
                results.get("seen", {}).get("ndcg@1", 0)
                - results.get("unseen", {}).get("ndcg@1", 0)
            )
            wandb.summary["run_duration_sec"] = time.perf_counter() - run_started
            output = {
                "config": config,
                "epochs_trained": len(history["train_loss"]),
                "best_epoch": history["best_epoch"],
                "stopped_early": history["stopped_early"],
                "best_val_loss": min(history["val_loss"]),
                "train_loss_history": history["train_loss"],
                "val_loss_history": history["val_loss"],
                "results": results,
            }
            output_path = output_dir / f"{run_name}.json"
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(output, output_file, indent=2)
            completed_runs.append({
                "run_name": run_name,
                "best_epoch": history["best_epoch"],
                "seen_ndcg1": results.get("seen", {}).get("ndcg@1", 0),
                "unseen_ndcg1": results.get("unseen", {}).get("ndcg@1", 0),
                "all_ndcg1": results.get("all", {}).get("ndcg@1", 0),
                "gap": results.get("seen", {}).get("ndcg@1", 0)
                - results.get("unseen", {}).get("ndcg@1", 0),
            })
            wandb.finish()
            run = None
            logger.info(
                "Run %d/%d: %s | best_epoch=%d | seen=%.3f unseen=%.3f all=%.3f",
                run_idx + 1,
                total_runs,
                run_name,
                history["best_epoch"],
                results.get("seen", {}).get("ndcg@1", 0),
                results.get("unseen", {}).get("ndcg@1", 0),
                results.get("all", {}).get("ndcg@1", 0),
            )
        except Exception:
            logger.exception("Run %d/%d failed: %s", run_idx + 1, total_runs, run_name)
            if run is not None:
                wandb.finish(exit_code=1)

    print("Run                              best_ep  seen_ndcg1  unseen_ndcg1  all_ndcg1  gap")
    for result in sorted(
        completed_runs,
        key=lambda item: float(item["unseen_ndcg1"]),
        reverse=True,
    ):
        print(
            f"{result['run_name']:<32} {result['best_epoch']:>7}"
            f"  {result['seen_ndcg1']:.3f}       {result['unseen_ndcg1']:.3f}"
            f"         {result['all_ndcg1']:.3f}    {result['gap']:.3f}"
        )


if __name__ == "__main__":
    main()
