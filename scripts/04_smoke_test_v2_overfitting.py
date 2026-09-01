"""Three-epoch smoke test for v2 train/test overfitting."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.data_loader import load_bpi2014_incidents
from src.evaluate import evaluate_retrieval
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.models import HeteroIncidentClassifier


NUM_EPOCHS = 3
BATCH_SIZE = 64
NUM_NEIGHBORS = [10, 10]
V1_GAP = 0.60


def _set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    data: HeteroData,
    mask: torch.Tensor,
    shuffle: bool,
) -> NeighborLoader:
    """Create a NeighborLoader for incident nodes selected by a mask."""
    return NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("incident", mask),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def _evaluate_mrr(
    model: HeteroIncidentClassifier,
    loader: NeighborLoader,
    device: torch.device,
) -> float:
    """Compute closure-code retrieval MRR over all loader seed incidents."""
    model.eval()
    ranks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_count = batch["incident"].batch_size
            logits = model(batch)[:seed_count]
            labels = batch["incident"].y[:seed_count]
            ordering = logits.argsort(dim=-1, descending=True)
            batch_ranks = (ordering == labels.unsqueeze(1)).float().argmax(dim=1) + 1
            ranks.append(batch_ranks.cpu())
    if not ranks:
        raise ValueError("evaluation mask selects no incidents")
    return evaluate_retrieval(torch.cat(ranks))["MRR"]


def _train_epoch(
    model: HeteroIncidentClassifier,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one sampled training epoch and return mean seed loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        seed_count = batch["incident"].batch_size
        logits = model(batch)[:seed_count]
        labels = batch["incident"].y[:seed_count]
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * seed_count
        total_examples += seed_count
    if total_examples == 0:
        raise ValueError("training mask selects no incidents")
    return total_loss / total_examples


def main() -> None:
    """Run three v2 epochs, print train/test MRR, and save smoke-test JSON."""
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    train_loader = _make_loader(data, masks["train_mask"], shuffle=True)
    train_eval_loader = _make_loader(data, masks["train_mask"], shuffle=False)
    test_loader = _make_loader(data, masks["test_mask"], shuffle=False)

    print("| Epoch | Train MRR | Test MRR | Gap |")
    print("|---:|---:|---:|---:|")
    epoch_results: list[dict[str, float | int]] = []
    for epoch in range(1, NUM_EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer, device)
        train_mrr = _evaluate_mrr(model, train_eval_loader, device)
        test_mrr = _evaluate_mrr(model, test_loader, device)
        gap = train_mrr - test_mrr
        epoch_results.append(
            {
                "epoch": epoch,
                "train_mrr": train_mrr,
                "test_mrr": test_mrr,
                "gap": gap,
            }
        )
        print(f"| {epoch} | {train_mrr:.3f} | {test_mrr:.3f} | {gap:.3f} |")

    v2_gap = float(epoch_results[-1]["gap"])
    reduced = v2_gap < V1_GAP
    print(f"\nv1 train/test gap: 0.86 - 0.26 = {V1_GAP:.2f}")
    print(f"v2 train/test gap: {v2_gap:.2f}")
    print(f"Did v2 reduce overfitting? {'YES' if reduced else 'NO'}")

    output_path = Path("results/smoke_test_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "v1_gap": V1_GAP,
                "v2_gap": v2_gap,
                "reduced_overfitting": reduced,
                "epochs": epoch_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
