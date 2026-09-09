"""Fuse text and GNN resolution retrieval signals."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

import wandb
from src.data_loader_sn import load_sn_incidents
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.models import HeteroIncidentClassifier

BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
EMBED_DIM = 384
HIDDEN_DIM = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
CI_DROPOUT_RATE = 0.5
ENTITY_EMBED_DIM = 64
RESULTS_DIR = Path("results/pathB")
ALPHAS = np.linspace(0.0, 1.0, 11)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def actual_device(device: str) -> torch.device:
    """Return requested device, falling back to CPU when CUDA is unavailable."""
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def node_counts(data: HeteroData) -> dict[str, int]:
    """Return non-incident node counts for entity embedding tables."""
    return {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }


def load_data(
    seed: int = 42,
) -> tuple[HeteroData, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Load filtered incidents, graph features, resolution targets, and masks."""
    set_seed(seed)
    all_incidents = load_sn_incidents()
    valid_rows = all_incidents["Closure Code"].notna().to_numpy()
    dataframe = all_incidents.loc[valid_rows].reset_index(drop=True)

    text_full = np.load("results/sn_text_embeddings.npy")
    if len(text_full) == len(all_incidents):
        text_embeddings = text_full[valid_rows]
    elif len(text_full) == len(dataframe):
        text_embeddings = text_full
    else:
        raise ValueError("text embeddings are not aligned with loaded incidents")

    graph = build_incident_graph_no_target(
        dataframe,
        add_assignment_group=True,
    )
    graph["incident"].x = torch.cat(
        [
            graph["incident"].x,
            torch.from_numpy(text_embeddings.astype(np.float32)),
        ],
        dim=1,
    )
    resolution_embeddings = torch.from_numpy(
        np.load("results/sn_resolution_embeddings.npy").astype(np.float32)
    )
    if len(resolution_embeddings) != len(dataframe):
        raise ValueError("resolution embeddings are not aligned with dataframe")

    masks = make_split_masks(dataframe, train_frac=0.7, val_frac=0.15)
    return (
        graph,
        resolution_embeddings,
        masks["train_mask"],
        masks["val_mask"],
        masks["test_mask"],
        dataframe,
    )


def compute_infonce_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute in-batch InfoNCE loss for predicted resolution embeddings."""
    predicted = F.normalize(predicted, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    logits = predicted @ target.T / temperature
    labels = torch.arange(len(predicted), device=predicted.device)
    return F.cross_entropy(logits, labels)


def evaluate_loss(
    model: HeteroIncidentClassifier,
    loader: NeighborLoader,
    targets: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> float:
    """Compute InfoNCE validation loss over loader seed incidents."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_count = batch["incident"].batch_size
            prediction = model(batch)[:seed_count]
            indices = batch["incident"].n_id[:seed_count]
            loss = compute_infonce_loss(prediction, targets[indices], temperature)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def train_infonce_gnn(
    data: HeteroData,
    resolution_embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    args: argparse.Namespace,
) -> HeteroIncidentClassifier:
    """Train InfoNCE GNN with a linear resolution head and early stopping."""
    set_seed(args.seed)
    device = actual_device(args.device)
    model = HeteroIncidentClassifier(
        data.metadata(),
        node_counts(data),
        input_dim=data["incident"].x.size(1),
        hidden_dim=HIDDEN_DIM,
        num_classes=EMBED_DIM,
        num_layers=2,
        dropout=DROPOUT,
        ci_dropout_rate=CI_DROPOUT_RATE,
        entity_embed_dim=ENTITY_EMBED_DIM,
    ).to(device)
    targets = resolution_embeddings.to(device)
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", train_mask),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", val_mask),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            seed_count = batch["incident"].batch_size
            prediction = model(batch)[:seed_count]
            indices = batch["incident"].n_id[:seed_count]
            loss = compute_infonce_loss(
                prediction, targets[indices], args.temperature
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(
            model, val_loader, targets, device, args.temperature
        )
        train_history.append(train_loss)
        val_history.append(val_loss)
        wandb.log(
            {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
            step=epoch,
        )
        print(
            f"GNN epoch {epoch:02d}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    model.training_history = {
        "epochs_trained": len(train_history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss": train_history,
        "val_loss": val_history,
    }
    return model


def extract_predictions(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    device: str,
) -> torch.Tensor:
    """Extract L2-normalized GNN resolution predictions in global row order."""
    target_device = actual_device(device)
    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=(
            "incident",
            torch.ones(data["incident"].num_nodes, dtype=torch.bool),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    predictions = torch.empty((data["incident"].num_nodes, EMBED_DIM))
    model.eval().to(target_device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(target_device)
            seed_count = batch["incident"].batch_size
            values = F.normalize(model(batch)[:seed_count], p=2, dim=1).cpu()
            indices = batch["incident"].n_id[:seed_count].cpu()
            predictions[indices] = values
    return predictions


def compute_similarity_matrices(
    text_embeddings: torch.Tensor | np.ndarray,
    predicted_res_embeddings: torch.Tensor | np.ndarray,
    true_res_embeddings: torch.Tensor | np.ndarray,
    train_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all-query versus train text and predicted-resolution cosine scores."""
    text = F.normalize(torch.as_tensor(text_embeddings).float(), p=2, dim=1)
    predicted = F.normalize(
        torch.as_tensor(predicted_res_embeddings).float(), p=2, dim=1
    )
    true = F.normalize(torch.as_tensor(true_res_embeddings).float(), p=2, dim=1)
    train_indices = torch.where(train_mask)[0]
    text_sim = text @ text[train_indices].T
    res_sim = predicted @ true[train_indices].T
    return text_sim.numpy(), res_sim.numpy()


def evaluate_hybrid(
    text_sim: np.ndarray,
    res_sim: np.ndarray,
    labels: np.ndarray,
    query_mask: torch.Tensor,
    train_mask: torch.Tensor,
    alpha: float,
) -> dict[str, float]:
    """Evaluate fused top-1/top-5 code matches and retrieved resolution cosine."""
    query_indices = torch.where(query_mask)[0].numpy()
    train_indices = torch.where(train_mask)[0].numpy()
    fused = alpha * res_sim[query_indices] + (1.0 - alpha) * text_sim[query_indices]
    ranking = np.argsort(-fused, axis=1)
    retrieved_labels = labels[train_indices][ranking]
    expected_labels = labels[query_indices]
    top1 = np.mean(retrieved_labels[:, 0] == expected_labels)
    top5 = np.mean(
        np.any(retrieved_labels[:, :5] == expected_labels[:, None], axis=1)
    )
    selected_res_cosine = res_sim[query_indices][
        np.arange(len(query_indices)), ranking[:, 0]
    ]
    return {
        "top1_match": float(top1),
        "top5_match": float(top5),
        "mean_cosine": float(np.mean(selected_res_cosine)),
    }


def sweep_alpha(
    text_sim: np.ndarray,
    res_sim: np.ndarray,
    labels: np.ndarray,
    query_mask: torch.Tensor,
    train_mask: torch.Tensor,
    alphas: list[float] | np.ndarray,
) -> dict[float, dict[str, float]]:
    """Evaluate each fusion alpha on query incidents and return its metrics."""
    return {
        float(alpha): evaluate_hybrid(
            text_sim, res_sim, labels, query_mask, train_mask, float(alpha)
        )
        for alpha in alphas
    }


def write_json(path: Path, value: object) -> None:
    """Write JSON output with readable stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    """Train GNN, tune fusion alpha on validation, and evaluate on test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    data, resolutions, train_mask, val_mask, test_mask, dataframe = load_data(
        args.seed
    )
    all_incidents = load_sn_incidents()
    valid_rows = all_incidents["Closure Code"].notna().to_numpy()
    text_full = np.load("results/sn_text_embeddings.npy")
    text_embeddings = (
        text_full[valid_rows] if len(text_full) == len(valid_rows) else text_full
    ).astype(np.float32)
    if len(text_embeddings) != len(dataframe):
        raise ValueError("text embeddings are not aligned with dataframe")

    labels = dataframe["Closure Code"].to_numpy()
    config = {
        "epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
        "patience": args.patience,
        "temperature": args.temperature,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "entity_embed_dim": ENTITY_EMBED_DIM,
        "ci_dropout_rate": CI_DROPOUT_RATE,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": 2,
        "dropout": DROPOUT,
        "features": "concat",
        "batch_size": BATCH_SIZE,
        "num_neighbors": NUM_NEIGHBORS,
    }
    run = wandb.init(
        project="gnn-incident-retrieval",
        name=f"resB_hybrid_e{args.epochs}_s{args.seed}",
        config=config,
    )
    try:
        model = train_infonce_gnn(
            data, resolutions, train_mask, val_mask, args
        )
        predictions = extract_predictions(model, data, args.device)
        text_sim, res_sim = compute_similarity_matrices(
            text_embeddings, predictions, resolutions, train_mask
        )

        val_metrics = sweep_alpha(
            text_sim, res_sim, labels, val_mask, train_mask, ALPHAS
        )
        print("\nValidation alpha sweep")
        print("alpha  top1_match  top5_match")
        for alpha in ALPHAS:
            metrics = val_metrics[float(alpha)]
            print(
                f"{alpha:.1f}    {metrics['top1_match']:.4f}       "
                f"{metrics['top5_match']:.4f}"
            )
        best_alpha = max(
            val_metrics,
            key=lambda alpha: (val_metrics[alpha]["top1_match"], alpha),
        )
        print(f"Best alpha: {best_alpha:.1f}")

        val_table = wandb.Table(columns=["alpha", "val_top1", "val_top5"])
        for alpha in ALPHAS:
            metrics = val_metrics[float(alpha)]
            val_table.add_data(
                float(alpha), metrics["top1_match"], metrics["top5_match"]
            )
        run.log({"alpha_sweep_val": val_table})

        test_results = {
            "text-only": evaluate_hybrid(
                text_sim, res_sim, labels, test_mask, train_mask, 0.0
            ),
            "gnn-only": evaluate_hybrid(
                text_sim, res_sim, labels, test_mask, train_mask, 1.0
            ),
            "hybrid": evaluate_hybrid(
                text_sim, res_sim, labels, test_mask, train_mask, best_alpha
            ),
        }
        print("\nTest comparison")
        print("model      top1_match  top5_match  mean_cosine")
        for name, metrics in test_results.items():
            print(
                f"{name:<10} {metrics['top1_match']:.4f}      "
                f"{metrics['top5_match']:.4f}      "
                f"{metrics['mean_cosine']:.4f}"
            )
        print(f"Best alpha: {best_alpha:.1f}")

        run.summary["best_alpha"] = best_alpha
        run.summary["test_top1"] = test_results["hybrid"]["top1_match"]
        run.summary["test_top5"] = test_results["hybrid"]["top5_match"]
        run.summary["test_mean_cosine"] = test_results["hybrid"]["mean_cosine"]
        history = model.training_history
        record = {
            "config": config,
            "alpha_sweep_val": [
                {
                    "alpha": float(alpha),
                    "top1_match": val_metrics[float(alpha)]["top1_match"],
                    "top5_match": val_metrics[float(alpha)]["top5_match"],
                }
                for alpha in ALPHAS
            ],
            "best_alpha": best_alpha,
            "test_results": test_results,
            "gnn_training": {
                "epochs_trained": history["epochs_trained"],
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
            },
        }
        write_json(
            RESULTS_DIR / f"hybrid_alpha_sweep_s{args.seed}.json",
            record,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
