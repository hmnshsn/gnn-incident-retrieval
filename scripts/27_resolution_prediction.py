"""Compare text, MLP, and GNN resolution-embedding retrieval."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.data_loader_sn import load_sn_incidents
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.models import HeteroIncidentClassifier
from src.resolution_loss import compute_resolution_loss

BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
EMBED_DIM = 384
HIDDEN_DIM = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
RESULTS_DIR = Path("results/pathB")


def write_json(path: Path, value: object) -> None:
    """Write JSON output with stable readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def compute_infonce_resolution_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE loss with each incident's resolution as its positive target.

    Args:
        predicted: Predicted embeddings with shape ``(batch_size, embed_dim)``.
        target: Target resolution embeddings with shape ``(batch_size, embed_dim)``.
        temperature: Temperature scaling factor for similarity logits.

    Returns:
        Scalar cross-entropy loss over in-batch resolution negatives.
    """
    predicted = F.normalize(predicted, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    logits = predicted @ target.T / temperature
    labels = torch.arange(len(predicted), device=predicted.device)
    return F.cross_entropy(logits, labels)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""
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


def load_data(
    seed: int = 42,
) -> tuple[HeteroData, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Load filtered incidents, concatenate features, and create temporal masks."""
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
        [graph["incident"].x, torch.from_numpy(text_embeddings.astype(np.float32))],
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


def node_counts(data: HeteroData) -> dict[str, int]:
    """Return non-incident node counts for classifier entity embeddings."""
    return {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }


def train_gnn_resolution(
    data: HeteroData,
    resolution_embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    seed: int,
    device: str,
    patience: int,
    loss_type: str,
    temperature: float,
    head_type: str,
    run: wandb.sdk.wandb_run.Run,
) -> tuple[HeteroIncidentClassifier, dict[str, object]]:
    """Train GNN with configured loss, projection head, and early stopping."""
    set_seed(seed)
    target_device = actual_device(device)
    model = HeteroIncidentClassifier(
        data.metadata(),
        node_counts(data),
        input_dim=data["incident"].x.size(1),
        hidden_dim=HIDDEN_DIM,
        num_classes=EMBED_DIM,
        num_layers=2,
        dropout=0.2,
        ci_dropout_rate=0.5,
        entity_embed_dim=64,
    ).to(target_device)
    if head_type == "proj":
        model.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
        ).to(target_device)
    targets = resolution_embeddings.to(target_device)
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
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    stopped_early = False
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(target_device)
            seed_count = batch["incident"].batch_size
            prediction = model(batch)[:seed_count]
            indices = batch["incident"].n_id[:seed_count]
            batch_targets = targets[indices]
            if loss_type == "infonce":
                loss = compute_infonce_resolution_loss(prediction, batch_targets, temperature)
            else:
                loss = compute_resolution_loss(prediction, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = evaluate_gnn_loss(
            model, val_loader, targets, target_device, loss_type, temperature
        )
        train_loss = float(np.mean(train_losses))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        run.log(
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
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
            stopped_early = True
            break
    model.load_state_dict(best_state)
    return model, {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_loss_history),
        "stopped_early": stopped_early,
    }


def evaluate_gnn_loss(
    model: HeteroIncidentClassifier,
    loader: NeighborLoader,
    targets: torch.Tensor,
    device: torch.device,
    loss_type: str,
    temperature: float,
) -> float:
    """Compute configured resolution loss over graph-loader seed nodes."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_count = batch["incident"].batch_size
            prediction = model(batch)[:seed_count]
            indices = batch["incident"].n_id[:seed_count]
            batch_targets = targets[indices]
            if loss_type == "infonce":
                loss = compute_infonce_resolution_loss(prediction, batch_targets, temperature)
            else:
                loss = compute_resolution_loss(prediction, batch_targets)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def train_mlp_resolution(
    features: np.ndarray | torch.Tensor,
    resolution_embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    seed: int,
    device: str,
    patience: int,
    loss_type: str,
    temperature: float,
    head_type: str,
    run: wandb.sdk.wandb_run.Run,
) -> tuple[nn.Sequential, dict[str, object]]:
    """Train MLP with configured loss, projection head, and early stopping."""
    set_seed(seed)
    target_device = actual_device(device)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    targets = resolution_embeddings.to(target_device)
    layers: list[nn.Module] = [
        nn.Linear(feature_tensor.size(1), HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(0.2),
    ]
    if head_type == "proj":
        layers.extend(
            [
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, EMBED_DIM),
            ]
        )
    else:
        layers.append(nn.Linear(HIDDEN_DIM, EMBED_DIM))
    model = nn.Sequential(*layers).to(target_device)
    train_dataset = TensorDataset(feature_tensor[train_mask], targets[train_mask])
    val_features = feature_tensor[val_mask].to(target_device)
    val_targets = targets[val_mask]
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    stopped_early = False
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_features, batch_targets in loader:
            prediction = model(batch_features.to(target_device))
            if loss_type == "infonce":
                loss = compute_infonce_resolution_loss(prediction, batch_targets, temperature)
            else:
                loss = compute_resolution_loss(prediction, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            if loss_type == "infonce":
                val_loss = compute_infonce_resolution_loss(
                    model(val_features), val_targets, temperature
                )
            else:
                val_loss = compute_resolution_loss(model(val_features), val_targets)
        train_loss = float(np.mean(losses))
        val_loss_value = float(val_loss.item())
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss_value)
        run.log(
            {"train_loss": train_loss, "val_loss": val_loss_value, "epoch": epoch},
            step=epoch,
        )
        print(
            f"MLP epoch {epoch:02d}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss_value:.6f}"
        )
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
            stopped_early = True
            break
    model.load_state_dict(best_state)
    return model, {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_loss_history),
        "stopped_early": stopped_early,
    }


def predict_gnn(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    device: str,
) -> torch.Tensor:
    """Extract normalized 384-dimensional GNN predictions in global order."""
    target_device = actual_device(device)
    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", torch.ones(data["incident"].num_nodes, dtype=torch.bool)),
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


def evaluate_resolution_retrieval(
    predicted_embeddings: torch.Tensor,
    true_resolution_embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    train_labels: Sequence[object],
    test_labels: Sequence[object],
    model_name: str,
    nearest_train_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate direct resolution quality and closure-code retrieval matches."""
    predicted = F.normalize(predicted_embeddings.float(), p=2, dim=1)
    true = F.normalize(true_resolution_embeddings.float(), p=2, dim=1)
    train_indices = torch.where(train_mask)[0]
    test_indices = torch.where(test_mask)[0]
    if nearest_train_indices is None:
        similarities = predicted[test_indices] @ true[train_indices].T
        ranking = similarities.argsort(dim=1, descending=True)
    else:
        ranking = torch.as_tensor(nearest_train_indices, device=train_indices.device)
        if ranking.ndim == 1:
            ranking = ranking[:, None]
    retrieved_labels = np.asarray(train_labels)[ranking[:, 0].cpu().numpy()]
    expected_labels = np.asarray(test_labels)
    top1 = float(np.mean(retrieved_labels == expected_labels))
    top5_labels = np.asarray(train_labels)[ranking[:, :5].cpu().numpy()]
    top5 = float(np.mean(np.any(top5_labels == expected_labels[:, None], axis=1)))
    target_similarity = (predicted[test_indices] * true[test_indices]).sum(dim=1)
    metrics = {
        "mean_cosine": float(target_similarity.mean().item()),
        "median_cosine": float(target_similarity.median().item()),
        "std_cosine": float(target_similarity.std(unbiased=False).item()),
        "top1_closure_match": top1,
        "top5_closure_match": top5,
    }
    print(f"\n{model_name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return metrics


def main() -> None:
    """Run text-only, MLP, and GNN resolution prediction comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss-type", choices=["cosine", "infonce"], default="cosine")
    parser.add_argument("--head-type", choices=["linear", "proj"], default="linear")
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    data, resolutions, train_mask, val_mask, test_mask, dataframe = load_data(args.seed)
    text_full = np.load("results/sn_text_embeddings.npy")
    valid_rows = load_sn_incidents()["Closure Code"].notna().to_numpy()
    text = torch.from_numpy(
        (text_full[valid_rows] if len(text_full) == len(valid_rows) else text_full).astype(np.float32)
    )
    train_labels = dataframe.loc[train_mask.numpy(), "Closure Code"].to_numpy()
    test_labels = dataframe.loc[test_mask.numpy(), "Closure Code"].to_numpy()
    text_predictions = torch.zeros_like(resolutions)
    train_text = F.normalize(text[train_mask], p=2, dim=1)
    text_similarity = F.normalize(text[test_mask], p=2, dim=1) @ train_text.T
    text_ranking = text_similarity.argsort(dim=1, descending=True)
    retrieved = torch.where(train_mask)[0][text_ranking[:, 0]]
    text_predictions[torch.where(test_mask)[0]] = resolutions[retrieved]
    results = {
        "Text-only": evaluate_resolution_retrieval(
            text_predictions, resolutions, train_mask, test_mask,
            train_labels, test_labels, "Text-only baseline",
            nearest_train_indices=text_ranking,
        )
    }
    write_json(RESULTS_DIR / "text_only_baseline.json", {
        "eval_metrics": results["Text-only"],
    })
    input_dim = int(data["incident"].x.size(1))
    config = {
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": args.patience,
        "loss_type": args.loss_type,
        "head_type": args.head_type,
        "temperature": args.temperature,
        "hidden_dim": HIDDEN_DIM,
        "dropout": 0.2,
        "features": "concat",
        "batch_size": BATCH_SIZE,
        "input_dim": input_dim,
    }

    mlp_run = wandb.init(
        project="gnn-incident-retrieval",
        name=f"resB_mlp_{args.loss_type}_{args.head_type}_e{args.epochs}_s{args.seed}",
        config={**config, "model_type": "mlp"},
    )
    try:
        mlp, mlp_history = train_mlp_resolution(
            data["incident"].x, resolutions, train_mask, val_mask,
            args.epochs, args.seed, args.device, args.patience,
            args.loss_type, args.temperature, args.head_type, mlp_run,
        )
        with torch.no_grad():
            mlp_predictions = F.normalize(
                mlp(data["incident"].x.to(actual_device(args.device))).cpu(), p=2, dim=1
            )
        results["MLP"] = evaluate_resolution_retrieval(
            mlp_predictions, resolutions, train_mask, test_mask,
            train_labels, test_labels, "MLP",
        )
        for key in (
            "mean_cosine", "median_cosine", "std_cosine",
            "top1_closure_match", "top5_closure_match",
        ):
            wandb.summary[key] = results["MLP"][key]
        for key in ("best_epoch", "best_val_loss", "epochs_trained"):
            wandb.summary[key] = mlp_history[key]
        mlp_record = {
            "config": {**config, "model_type": "mlp"},
            "epochs_trained": mlp_history["epochs_trained"],
            "best_epoch": mlp_history["best_epoch"],
            "best_val_loss": mlp_history["best_val_loss"],
            "train_loss_history": mlp_history["train_loss"],
            "val_loss_history": mlp_history["val_loss"],
            "eval_metrics": results["MLP"],
        }
        write_json(
            RESULTS_DIR / f"resB_mlp_{args.loss_type}_{args.head_type}_e{args.epochs}_s{args.seed}.json",
            mlp_record,
        )
    finally:
        wandb.finish()

    gnn_config = {
        **config,
        "model_type": "gnn",
        "entity_embed_dim": 64,
        "ci_dropout_rate": 0.5,
        "num_layers": 2,
        "num_neighbors": NUM_NEIGHBORS,
    }
    gnn_run = wandb.init(
        project="gnn-incident-retrieval",
        name=f"resB_gnn_{args.loss_type}_{args.head_type}_e{args.epochs}_s{args.seed}",
        config=gnn_config,
    )
    try:
        gnn, gnn_history = train_gnn_resolution(
            data, resolutions, train_mask, val_mask,
            args.epochs, args.seed, args.device, args.patience,
            args.loss_type, args.temperature, args.head_type, gnn_run,
        )
        gnn_predictions = predict_gnn(gnn, data, args.device)
        results["GNN"] = evaluate_resolution_retrieval(
            gnn_predictions, resolutions, train_mask, test_mask,
            train_labels, test_labels, "GNN",
        )
        for key in (
            "mean_cosine", "median_cosine", "std_cosine",
            "top1_closure_match", "top5_closure_match",
        ):
            wandb.summary[key] = results["GNN"][key]
        for key in ("best_epoch", "best_val_loss", "epochs_trained"):
            wandb.summary[key] = gnn_history[key]
        gnn_record = {
            "config": gnn_config,
            "epochs_trained": gnn_history["epochs_trained"],
            "best_epoch": gnn_history["best_epoch"],
            "best_val_loss": gnn_history["best_val_loss"],
            "train_loss_history": gnn_history["train_loss"],
            "val_loss_history": gnn_history["val_loss"],
            "eval_metrics": results["GNN"],
        }
        write_json(
            RESULTS_DIR / f"resB_gnn_{args.loss_type}_{args.head_type}_e{args.epochs}_s{args.seed}.json",
            gnn_record,
        )
    finally:
        wandb.finish()

    print("\nComparison")
    print("model       mean_cosine  top1_match  top5_match")
    for name, metrics in results.items():
        print(
            f"{name:<11} {metrics['mean_cosine']:.4f}       "
            f"{metrics['top1_closure_match']:.4f}      "
            f"{metrics['top5_closure_match']:.4f}"
        )


if __name__ == "__main__":
    main()
