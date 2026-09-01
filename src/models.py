"""Minibatch heterogeneous incident classifier for the v2 experiment."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from tqdm import tqdm


class HeteroIncidentClassifier(nn.Module):
    """Heterogeneous GraphSAGE classifier for incident closure codes."""

    def __init__(
        self,
        metadata: tuple[Sequence[str], Sequence[tuple[str, str, str]]],
        node_counts: Mapping[str, int],
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 14,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialize entity embeddings, message-passing layers, and head.

        Args:
            metadata: PyG metadata tuple ``(node_types, edge_types)``.
            node_counts: Counts for available non-incident node types.
            input_dim: Width of all graph input features.
            hidden_dim: Width of each hidden representation.
            num_classes: Number of closure-code classes.
            num_layers: Number of heterogeneous GraphSAGE layers.
            dropout: Dropout probability between message-passing layers.
        """
        super().__init__()
        node_types, edge_types = metadata
        if input_dim < 1 or hidden_dim < 1 or num_classes < 1 or num_layers < 1:
            raise ValueError("model dimensions and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.node_types = tuple(node_types)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.entity_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, input_dim)
                for node_type, count in node_counts.items()
                if node_type != "incident" and count > 0
            }
        )
        self.convs = nn.ModuleList()
        for layer_index in range(num_layers):
            channels = input_dim if layer_index == 0 else hidden_dim
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((channels, channels), hidden_dim)
                        for edge_type in edge_types
                    },
                    aggr="mean",
                )
            )
        self.initial_residuals = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dim, hidden_dim)
                for node_type in node_types
            }
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Compute raw closure-code logits for all sampled incident nodes.

        Args:
            data: Full or NeighborLoader-sampled heterogeneous graph. Sampled
                entity ``n_id`` values are used to index global embeddings.

        Returns:
            Tensor of shape ``[number_of_sampled_incidents, num_classes]``.
        """
        features: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            if node_type == "incident":
                features[node_type] = data[node_type].x.detach()
                continue
            if node_type in self.entity_embeddings:
                store = data[node_type]
                if hasattr(store, "n_id"):
                    node_indices = store.n_id.to(
                        self.entity_embeddings[node_type].weight.device
                    )
                else:
                    node_indices = torch.arange(
                        store.num_nodes,
                        device=self.entity_embeddings[node_type].weight.device,
                    )
                features[node_type] = self.entity_embeddings[node_type](node_indices)
            elif hasattr(data[node_type], "x"):
                features[node_type] = data[node_type].x.detach()
            else:
                raise KeyError(f"No features or embedding for node type {node_type}")

        hidden = features
        for layer_index, conv in enumerate(self.convs):
            message_outputs = conv(hidden, data.edge_index_dict)
            if layer_index == 0:
                residuals = {
                    node_type: self.initial_residuals[node_type](node_features)
                    for node_type, node_features in hidden.items()
                }
            else:
                residuals = hidden
            hidden = {
                node_type: self.dropout(
                    F.relu(message_outputs.get(node_type, residuals[node_type]))
                )
                for node_type in self.node_types
            }
        return self.classifier(hidden["incident"])


def _neighbor_loader(
    data: HeteroData,
    mask: torch.Tensor,
    batch_size: int,
    num_neighbors: Sequence[int],
    shuffle: bool,
) -> NeighborLoader:
    """Create a NeighborLoader over incident nodes selected by a mask."""
    return NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        input_nodes=("incident", mask.cpu()),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _evaluate_loader(
    model: HeteroIncidentClassifier,
    loader: NeighborLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute mean cross-entropy and accuracy over a sampled loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_count = batch["incident"].batch_size
            logits = model(batch)[:seed_count]
            labels = batch["incident"].y[:seed_count]
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * seed_count
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_examples += seed_count
    if total_examples == 0:
        raise ValueError("evaluation mask selects no incidents")
    return total_loss / total_examples, total_correct / total_examples


def train_minibatch(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.005,
    batch_size: int = 1024,
    num_neighbors: Sequence[int] | None = None,
    device: str = "cuda:0",
) -> tuple[HeteroIncidentClassifier, dict[str, list[float]]]:
    """Train the classifier with sampled incident minibatches.

    Args:
        model: Heterogeneous incident classifier.
        data: Target-free heterogeneous graph on CPU.
        train_mask: Boolean mask selecting training incidents.
        val_mask: Boolean mask selecting validation incidents.
        epochs: Number of training epochs.
        lr: AdamW learning rate.
        batch_size: Number of seed incidents per sampled batch.
        num_neighbors: Neighbors sampled per hop; defaults to ``[15, 10]``.
        device: Requested torch device, falling back to CPU if unavailable.

    Returns:
        The model restored to its best validation-loss state and a history
        dictionary containing ``train_loss``, ``val_loss`` and ``val_acc``.
    """
    if epochs < 1 or lr <= 0 or batch_size < 1:
        raise ValueError("epochs, lr, and batch_size must be positive")
    neighbors = [15, 10] if num_neighbors is None else list(num_neighbors)
    if not neighbors or any(value < 0 for value in neighbors):
        raise ValueError("num_neighbors must contain non-negative values")
    requested_device = torch.device(device)
    actual_device = (
        requested_device
        if requested_device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(actual_device)
    train_loader = _neighbor_loader(
        data, train_mask, batch_size, neighbors, shuffle=True
    )
    val_loader = _neighbor_loader(data, val_mask, batch_size, neighbors, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    epoch_bar = tqdm(range(1, epochs + 1), total=epochs, desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        total_train_loss = 0.0
        total_examples = 0
        batch_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            leave=False,
        )
        for batch in batch_bar:
            batch = batch.to(actual_device)
            seed_count = batch["incident"].batch_size
            logits = model(batch)[:seed_count]
            labels = batch["incident"].y[:seed_count]
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += float(loss.detach().item()) * seed_count
            total_examples += seed_count
        if total_examples == 0:
            raise ValueError("train_mask selects no incidents")
        train_loss = total_train_loss / total_examples
        val_loss, val_acc = _evaluate_loader(model, val_loader, actual_device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

    model.load_state_dict(best_state)
    return model, history


def predict_ranks(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    eval_mask: torch.Tensor,
    device: str = "cuda:0",
    batch_size: int = 1024,
    num_neighbors: Sequence[int] | None = None,
) -> torch.Tensor:
    """Rank each selected incident's true closure code from classifier logits.

    Args:
        model: Trained heterogeneous incident classifier.
        data: Target-free heterogeneous graph on CPU.
        eval_mask: Boolean mask selecting incidents to rank.
        device: Requested torch device, falling back to CPU if unavailable.
        batch_size: Number of seed incidents per sampled batch.
        num_neighbors: Neighbors sampled per hop; defaults to ``[15, 10]``.

    Returns:
        FloatTensor of one-indexed ranks ordered by the selected mask.
    """
    neighbors = [15, 10] if num_neighbors is None else list(num_neighbors)
    requested_device = torch.device(device)
    actual_device = (
        requested_device
        if requested_device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    loader = _neighbor_loader(
        data, eval_mask, batch_size, neighbors, shuffle=False
    )
    model = model.to(actual_device)
    model.eval()
    ranks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(actual_device)
            seed_count = batch["incident"].batch_size
            logits = model(batch)[:seed_count]
            labels = batch["incident"].y[:seed_count]
            ordering = logits.argsort(dim=-1, descending=True)
            rank = (ordering == labels.unsqueeze(1)).float().argmax(dim=1) + 1
            ranks.append(rank.cpu())
    if not ranks:
        raise ValueError("eval_mask selects no incidents")
    return torch.cat(ranks).to(torch.float32)
