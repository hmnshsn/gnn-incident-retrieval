"""Minibatch heterogeneous incident classifier for the v2 experiment."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from tqdm import tqdm

from src.infonce_loss import compute_infonce_loss
from src.triplet_loss import batch_triplet_loss


logger = logging.getLogger(__name__)


class HeteroIncidentClassifier(nn.Module):
    """Heterogeneous GraphSAGE classifier with configurable entity embeddings."""

    def __init__(
        self,
        metadata: tuple[Sequence[str], Sequence[tuple[str, str, str]]],
        node_counts: Mapping[str, int],
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 14,
        num_layers: int = 2,
        dropout: float = 0.3,
        ci_dropout_rate: float = 0.0,
        ci_dropout_mode: str = "embedding",
        ci_feature_dim: int | None = None,
        entity_embed_dim: int | None = None,
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
            ci_dropout_rate: Probability of dropping CI embeddings, CI edges,
                or both during training.
            ci_dropout_mode: CI dropout strategy: ``embedding``, ``edge``, or
                ``both``.
            ci_feature_dim: Width of optional fixed inductive CI features.
            entity_embed_dim: Width of learnable entity embeddings. Defaults to
                ``input_dim``.
        """
        super().__init__()
        node_types, edge_types = metadata
        if input_dim < 1 or hidden_dim < 1 or num_classes < 1 or num_layers < 1:
            raise ValueError("model dimensions and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= ci_dropout_rate <= 1:
            raise ValueError("ci_dropout_rate must be in [0, 1]")
        if ci_dropout_mode not in {"embedding", "edge", "both"}:
            raise ValueError("ci_dropout_mode must be embedding, edge, or both")
        if ci_feature_dim is not None and ci_feature_dim < 1:
            raise ValueError("ci_feature_dim must be positive")
        if entity_embed_dim is not None and entity_embed_dim < 1:
            raise ValueError("entity_embed_dim must be positive")

        self.node_types = tuple(node_types)
        self.ci_dropout_rate = ci_dropout_rate
        self.ci_dropout_mode = ci_dropout_mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = input_dim
        _embed_dim = (
            entity_embed_dim if entity_embed_dim is not None else input_dim
        )
        self._entity_embed_dim = _embed_dim
        self.entity_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, _embed_dim)
                for node_type, count in node_counts.items()
                if node_type != "incident" and count > 0
            }
        )
        if _embed_dim != input_dim:
            self.entity_projections = nn.ModuleDict(
                {
                    node_type: nn.Linear(_embed_dim, input_dim)
                    for node_type in self.entity_embeddings
                }
            )
        else:
            self.entity_projections = None
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
        self.ci_feature_proj = (
            nn.Linear(ci_feature_dim, _embed_dim)
            if ci_feature_dim is not None
            else None
        )
        self.ci_both_proj = (
            nn.Linear(_embed_dim + _embed_dim, _embed_dim)
            if ci_feature_dim is not None
            else None
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _apply_ci_dropout(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
        """Apply configured CI embedding and/or exact-CI edge dropout."""
        if not self.training or self.ci_dropout_rate <= 0:
            return x_dict, edge_index_dict

        dropped_x = dict(x_dict)
        dropped_edges = dict(edge_index_dict)
        rate = self.ci_dropout_rate
        if self.ci_dropout_mode in {"embedding", "both"} and "ci" in dropped_x:
            mask = (
                torch.rand(
                    dropped_x["ci"].size(0),
                    device=dropped_x["ci"].device,
                )
                >= rate
            ).to(dropped_x["ci"].dtype)
            dropped_x["ci"] = dropped_x["ci"] * mask.unsqueeze(-1)
        if self.ci_dropout_mode in {"edge", "both"}:
            for edge_type, edge_index in dropped_edges.items():
                if edge_type[0] != "ci" and edge_type[2] != "ci":
                    continue
                keep = torch.rand(
                    edge_index.size(1),
                    device=edge_index.device,
                ) >= rate
                dropped_edges[edge_type] = edge_index[:, keep]
        return dropped_x, dropped_edges

    def forward(
        self,
        data: HeteroData,
        ci_input_features: torch.Tensor | None = None,
        ci_features_mode: str = "learnable",
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute closure-code logits using learnable or inductive CI features.

        Args:
            data: Full or NeighborLoader-sampled heterogeneous graph.
            ci_input_features: Optional CI features aligned to sampled CI nodes.
            ci_features_mode: ``learnable``, ``inductive``, or ``both``.
            return_embeddings: Whether to return pre-classifier incident embeddings.
        """

        features: dict[str, torch.Tensor] = {}
        if ci_features_mode not in {"learnable", "inductive", "both"}:
            raise ValueError(
                "ci_features_mode must be learnable, inductive, or both"
            )
        if ci_features_mode != "learnable" and ci_input_features is None:
            raise ValueError("ci_input_features required for inductive CI features")
        if ci_features_mode == "both" and self.ci_both_proj is None:
            raise ValueError("ci_feature_dim required for both CI features")
        for node_type in self.node_types:
            if node_type == "incident":
                features[node_type] = data[node_type].x.detach()
                continue
            store = data[node_type]
            projected = None
            if node_type == "ci" and ci_features_mode != "learnable":
                projected = (
                    self.ci_feature_proj(ci_input_features)
                    if self.ci_feature_proj is not None
                    else ci_input_features
                )
            if node_type in self.entity_embeddings:
                if hasattr(store, "n_id"):
                    node_indices = store.n_id.to(
                        self.entity_embeddings[node_type].weight.device
                    )
                else:
                    node_indices = torch.arange(
                        store.num_nodes,
                        device=self.entity_embeddings[node_type].weight.device,
                    )
                embedding = self.entity_embeddings[node_type](node_indices)
                if node_type == "ci" and ci_features_mode == "inductive":
                    raw = projected
                elif node_type == "ci" and ci_features_mode == "both":
                    raw = self.ci_both_proj(
                        torch.cat([projected, embedding], dim=-1)
                    )
                else:
                    raw = embedding
            elif hasattr(store, "x"):
                raw = store.x.detach()
            else:
                raise KeyError(f"No features or embedding for node type {node_type}")
            if self.entity_projections is not None and node_type in self.entity_projections:
                features[node_type] = self.entity_projections[node_type](raw)
            else:
                features[node_type] = raw

        hidden, edge_index_dict = self._apply_ci_dropout(
            features,
            data.edge_index_dict,
        )
        for layer_index, conv in enumerate(self.convs):
            message_outputs = conv(hidden, edge_index_dict)
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
        incident_embeddings = hidden["incident"]
        logits = self.classifier(incident_embeddings)
        return (logits, incident_embeddings) if return_embeddings else logits


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
    ci_input_features: torch.Tensor | None = None,
    ci_features_mode: str = "learnable",
    triplet_weight: float = 0.0,
    triplet_margin: float = 0.2,
    contrastive_loss: str = "triplet",
    temperature: float = 0.07,
) -> tuple[float, float, float, int]:
    """Compute validation loss, accuracy, and optional contrastive metrics."""
    model.eval()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_active_triplets = 0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_count = batch["incident"].batch_size
            batch_ci_features = (
                ci_input_features[batch["ci"].n_id.cpu()].to(device)
                if ci_input_features is not None
                else None
            )
            if triplet_weight > 0:
                logits, embeddings = model(
                    batch, batch_ci_features, ci_features_mode,
                    return_embeddings=True,
                )
                embeddings = embeddings[:seed_count]
            else:
                logits = model(batch, batch_ci_features, ci_features_mode)
            logits = logits[:seed_count]
            labels = batch["incident"].y[:seed_count]
            ce_loss = F.cross_entropy(logits, labels)
            if triplet_weight > 0:
                if contrastive_loss == "triplet":
                    contrastive_value, n_active = batch_triplet_loss(
                        embeddings, labels, margin=triplet_margin
                    )
                else:
                    contrastive_value = compute_infonce_loss(
                        embeddings, labels, temperature=temperature
                    )
                    n_active = int(
                        (
                            (labels[:, None] == labels[None, :])
                            & ~torch.eye(
                                seed_count,
                                dtype=torch.bool,
                                device=labels.device,
                            )
                        ).any(dim=1).sum().item()
                    )
            else:
                contrastive_value, n_active = ce_loss.new_zeros(()), 0
            loss = (1.0 - triplet_weight) * ce_loss + triplet_weight * contrastive_value
            total_loss += float(loss.item()) * seed_count
            total_triplet_loss += float(contrastive_value.item()) * seed_count
            total_active_triplets += n_active
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_examples += seed_count
    if total_examples == 0:
        raise ValueError("evaluation mask selects no incidents")
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        total_triplet_loss / total_examples,
        total_active_triplets,
    )


def train_minibatch(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    num_neighbors: Sequence[int] | None = None,
    device: str = "cuda:0",
    patience: int = 10,
    min_delta: float = 1e-4,
    ci_input_features: torch.Tensor | None = None,
    ci_features_mode: str = "learnable",
    triplet_weight: float = 0.0,
    triplet_margin: float = 0.2,
    contrastive_loss: str = "triplet",
    temperature: float = 0.07,
) -> tuple[HeteroIncidentClassifier, dict[str, object]]:
    """Train the classifier with sampled incident minibatches.

    Args:
        model: Heterogeneous incident classifier.
        data: Target-free heterogeneous graph on CPU.
        train_mask: Boolean mask selecting training incidents.
        val_mask: Boolean mask selecting validation incidents.
        epochs: Number of training epochs.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay coefficient.
        batch_size: Number of seed incidents per sampled batch.
        num_neighbors: Neighbors sampled per hop; defaults to ``[15, 10]``.
        device: Requested torch device, falling back to CPU if unavailable.
        patience: Number of consecutive non-improving epochs before stopping.
        min_delta: Minimum validation-loss decrease counted as improvement.
        ci_input_features: Full CI feature matrix indexed by global node ID.
        ci_features_mode: CI feature assembly mode.
        triplet_weight: Weight of contrastive loss relative to cross-entropy.
        triplet_margin: Margin used by semi-hard triplet mining.
        contrastive_loss: Contrastive loss type, ``triplet`` or ``infonce``.
        temperature: InfoNCE temperature.

    Returns:
        The model restored to its best validation-loss state and a history
        dictionary containing losses, validation accuracy, ``best_epoch``, and
        ``stopped_early``.
    """
    if epochs < 1 or lr <= 0 or batch_size < 1:
        raise ValueError("epochs, lr, and batch_size must be positive")
    if patience < 1 or min_delta < 0:
        raise ValueError("patience must be positive and min_delta non-negative")
    if not 0 <= triplet_weight <= 1 or triplet_margin < 0:
        raise ValueError("triplet_weight must be in [0, 1] and triplet_margin non-negative")
    if contrastive_loss not in {"triplet", "infonce"}:
        raise ValueError("contrastive_loss must be triplet or infonce")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []
    train_triplet_history: list[float] = []
    val_triplet_history: list[float] = []
    history: dict[str, object] = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_acc": val_acc_history,
        "best_epoch": 0,
        "stopped_early": False,
    }
    if triplet_weight > 0:
        history["train_triplet_loss"] = train_triplet_history
        history["val_triplet_loss"] = val_triplet_history
    try:
        import wandb
    except ImportError:
        wandb = None
    best_val_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_counter = 0

    epoch_bar = tqdm(range(1, epochs + 1), total=epochs, desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        total_train_loss = 0.0
        total_train_triplet_loss = 0.0
        total_active_triplets = 0
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
            batch_ci_features = (
                ci_input_features[batch["ci"].n_id.cpu()].to(actual_device)
                if ci_input_features is not None
                else None
            )
            if triplet_weight > 0:
                logits, embeddings = model(
                    batch, batch_ci_features, ci_features_mode,
                    return_embeddings=True,
                )
                embeddings = embeddings[:seed_count]
            else:
                logits = model(batch, batch_ci_features, ci_features_mode)
            logits = logits[:seed_count]
            labels = batch["incident"].y[:seed_count]
            ce_loss = F.cross_entropy(logits, labels)
            if triplet_weight > 0:
                if contrastive_loss == "triplet":
                    contrastive_value, n_active = batch_triplet_loss(
                        embeddings, labels, margin=triplet_margin
                    )
                else:
                    contrastive_value = compute_infonce_loss(
                        embeddings, labels, temperature=temperature
                    )
                    n_active = int(
                        (
                            (labels[:, None] == labels[None, :])
                            & ~torch.eye(
                                seed_count,
                                dtype=torch.bool,
                                device=labels.device,
                            )
                        ).any(dim=1).sum().item()
                    )
            else:
                contrastive_value, n_active = ce_loss.new_zeros(()), 0
            loss = (1.0 - triplet_weight) * ce_loss + triplet_weight * contrastive_value
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += float(loss.detach().item()) * seed_count
            total_train_triplet_loss += float(contrastive_value.detach().item()) * seed_count
            total_active_triplets += n_active
            total_examples += seed_count
        if total_examples == 0:
            raise ValueError("train_mask selects no incidents")
        train_loss = total_train_loss / total_examples
        train_triplet_loss = total_train_triplet_loss / total_examples
        val_loss, val_acc, val_triplet_loss, val_active_triplets = _evaluate_loader(
            model,
            val_loader,
            actual_device,
            ci_input_features,
            ci_features_mode,
            triplet_weight,
            triplet_margin,
            contrastive_loss,
            temperature,
        )
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if triplet_weight > 0:
            train_triplet_history.append(train_triplet_loss)
            val_triplet_history.append(val_triplet_loss)
            epoch_metrics.update(
                {
                    "train_triplet_loss": train_triplet_loss,
                    "val_triplet_loss": val_triplet_loss,
                    "n_active_triplets": total_active_triplets + val_active_triplets,
                }
            )
        if wandb is not None and wandb.run is not None:
            wandb.log(epoch_metrics, step=epoch)
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )
        if patience_counter >= patience:
            logger.info(
                f"Early stopping at epoch {epoch}, best epoch {best_epoch}"
            )
            history["stopped_early"] = True
            break

    if best_weights is None:
        raise RuntimeError("No best model weights were saved")
    history["best_epoch"] = best_epoch
    model.load_state_dict(best_weights)
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


def extract_incident_embeddings(
    model: HeteroIncidentClassifier,
    data: HeteroData,
    loader: NeighborLoader,
    device: str = "cpu",
    ci_input_features: torch.Tensor | None = None,
    ci_features_mode: str = "learnable",
) -> torch.Tensor:
    """Extract pre-classifier incident embeddings in global node order.

    Runs inference over all minibatches, captures the hidden representation
    immediately before the classifier head for seed incident nodes only, and
    maps sampled local nodes back to global incident indices using
    ``batch['incident'].n_id``. Revisited nodes keep their first observed
    embedding.

    Args:
        model: Trained HeteroIncidentClassifier instance.
        data: Full HeteroData graph object.
        loader: NeighborLoader configured for incident nodes to extract.
        device: Device string used for inference.

    Returns:
        Tensor of shape ``(num_incidents, hidden_dim)`` ordered by global index.
    """
    actual_device = torch.device(device)
    if actual_device.type == "cuda" and not torch.cuda.is_available():
        actual_device = torch.device("cpu")
    model = model.to(actual_device)
    model.eval()
    captured: list[torch.Tensor] = []
    embeddings_by_index: dict[int, torch.Tensor] = {}
    hook = model.classifier.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0])
    )
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(actual_device)
                captured.clear()
                batch_ci_features = (
                    ci_input_features[batch["ci"].n_id.cpu()].to(actual_device)
                    if ci_input_features is not None
                    else None
                )
                model(batch, batch_ci_features, ci_features_mode)
                if not captured:
                    raise RuntimeError("classifier hook did not capture embeddings")
                incident_store = batch["incident"]
                seed_count = incident_store.batch_size
                hidden = captured[0][:seed_count].detach().cpu()
                if hasattr(incident_store, "n_id"):
                    global_indices = incident_store.n_id[:seed_count].detach().cpu().tolist()
                else:
                    global_indices = list(range(seed_count))
                for global_index, embedding in zip(global_indices, hidden):
                    embeddings_by_index.setdefault(global_index, embedding)
    finally:
        hook.remove()

    if not embeddings_by_index:
        raise ValueError("loader yielded no incident embeddings")
    hidden_dim = next(iter(embeddings_by_index.values())).numel()
    embeddings = torch.zeros(data["incident"].num_nodes, hidden_dim)
    for global_index, embedding in embeddings_by_index.items():
        embeddings[global_index] = embedding
    return embeddings
