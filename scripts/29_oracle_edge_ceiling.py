"""Oracle edge ceiling: measure skyline performance with perfect neighbour selection.

WARNING: This script DELIBERATELY uses relevance labels to choose edges at inference
time and is NOT a deployable method. Results represent an upper bound on what could be
achieved with perfect neighbour selection, not a real retrieval system.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.knn_edges import add_knn_relation, build_text_knn_edges
from src.logging_config import setup_logging
from src.models import HeteroIncidentClassifier, extract_incident_embeddings, train_minibatch
from src.relevance import compute_relevance_matrix
from src.sweep_runner import (
    _build_node_counts,
    _evaluate_group,
    _prepare_dataframe,
    _set_seed,
    _split_positions,
    _training_ci_indices,
    build_graph_for_features,
)
from torch_geometric.loader import NeighborLoader


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]


def _neighbor_loader(
    data: HeteroData,
    mask: np.ndarray,
    batch_size: int,
    num_neighbors: list[int],
    shuffle: bool,
) -> NeighborLoader:
    """Build a NeighborLoader for incident nodes selected by mask."""
    mask_tensor = torch.zeros(data["incident"].num_nodes, dtype=torch.bool)
    mask_tensor[mask] = True
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=("incident", mask_tensor),
        shuffle=shuffle,
    )


def build_oracle_edges(
    text_embeddings: np.ndarray,
    dataframe: object,
    unseen_test_indices: np.ndarray,
    train_indices: np.ndarray,
    oracle_k: int,
) -> torch.Tensor:
    """Build oracle edges from unseen test incidents to top-k training by relevance."""
    relevance = compute_relevance_matrix(dataframe, train_indices, unseen_test_indices)
    relevance_array = relevance.toarray()
    text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_normalized = text_embeddings / np.maximum(text_norms, 1e-12)
    query_embeddings = text_normalized[unseen_test_indices]
    train_embeddings = text_normalized[train_indices]
    similarities = query_embeddings @ train_embeddings.T
    edge_sources = []
    edge_targets = []
    for i, query_idx in enumerate(unseen_test_indices):
        relevance_row = relevance_array[i]
        similarity_row = similarities[i]
        combined_scores = relevance_row * 1e6 + similarity_row
        top_k_positions = np.argpartition(-combined_scores, min(oracle_k, len(train_indices)) - 1)[
            :oracle_k
        ]
        top_k_positions = top_k_positions[np.argsort(-combined_scores[top_k_positions])]
        edge_sources.extend([query_idx] * len(top_k_positions))
        edge_targets.extend(train_indices[top_k_positions].tolist())
    return torch.tensor(np.stack([edge_sources, edge_targets]), dtype=torch.long)


def replace_unseen_edges(
    data: HeteroData,
    oracle_edges: torch.Tensor,
    unseen_test_indices: np.ndarray,
) -> HeteroData:
    """Replace text_similar edges from unseen test incidents with oracle edges."""
    oracle_data = data.clone()
    forward_key = ("incident", "text_similar", "incident")
    reverse_key = ("incident", "rev_text_similar", "incident")
    original_forward = data[forward_key].edge_index
    original_reverse = data[reverse_key].edge_index
    unseen_set = set(unseen_test_indices.tolist())
    forward_mask = torch.tensor(
        [int(src) not in unseen_set for src in original_forward[0].tolist()],
        dtype=torch.bool,
    )
    reverse_mask = torch.tensor(
        [int(tgt) not in unseen_set for tgt in original_reverse[1].tolist()],
        dtype=torch.bool,
    )
    kept_forward = original_forward[:, forward_mask]
    kept_reverse = original_reverse[:, reverse_mask]
    new_forward = torch.cat([kept_forward, oracle_edges], dim=1)
    new_reverse = torch.cat([kept_reverse, oracle_edges.flip(0)], dim=1)
    oracle_data[forward_key].edge_index = new_forward
    oracle_data[reverse_key].edge_index = new_reverse
    return oracle_data


def run_single_seed(
    seed: int,
    args: argparse.Namespace,
    dataframe: object,
    text_embeddings: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    test_seen: np.ndarray,
    test_unseen: np.ndarray,
    logger: logging.Logger,
) -> dict:
    """Train model and evaluate on control and oracle graphs for one seed."""
    _set_seed(seed)
    logger.info("Seed %d: building graph with text-kNN k=%d", seed, args.knn_k)
    data = build_graph_for_features(args.features, dataframe, text_embeddings)
    knn_edges, knn_stats = build_text_knn_edges(
        text_embeddings,
        range(len(dataframe)),
        train_indices,
        args.knn_k,
    )
    add_knn_relation(data, knn_edges, "text_similar")
    logger.info(
        "Seed %d: added %d text-kNN edges, mean similarity %.3f",
        seed, knn_stats["num_edges"], knn_stats["mean_similarity"],
    )
    train_mask = np.zeros(len(dataframe), dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.zeros(len(dataframe), dtype=bool)
    val_mask[val_indices] = True
    data["incident"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data["incident"].val_mask = torch.tensor(val_mask, dtype=torch.bool)
    node_counts = _build_node_counts(data)
    input_dim = data["incident"].x.size(1)
    model = HeteroIncidentClassifier(
        metadata=data.metadata(),
        node_counts=node_counts,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=int(dataframe["Closure Code"].nunique()),
        num_layers=args.num_layers,
        dropout=args.dropout,
        ci_dropout_rate=args.ci_dropout_rate,
        ci_dropout_mode=args.ci_dropout_mode,
        entity_embed_dim=args.entity_embed_dim,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Seed %d: training for up to %d epochs", seed, args.epochs)
    model, history = train_minibatch(
        model,
        data,
        data["incident"].train_mask,
        data["incident"].val_mask,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        device=device,
    )
    logger.info(
        "Seed %d: training complete, best epoch %d",
        seed, history["best_epoch"],
    )
    all_mask = np.ones(len(dataframe), dtype=bool)
    loader = _neighbor_loader(data, all_mask, BATCH_SIZE, NUM_NEIGHBORS, shuffle=False)
    control_embeddings = extract_incident_embeddings(model, data, loader, device=device).cpu().numpy()
    control_metrics = {}
    for group_name, group_indices in [
        ("seen", test_seen),
        ("unseen", test_unseen),
        ("all", test_indices),
    ]:
        control_metrics[group_name] = _evaluate_group(
            control_embeddings, dataframe, train_indices, group_indices
        )
    oracle_results = {}
    for oracle_k in args.oracle_k:
        logger.info("Seed %d: building oracle graph with k=%d", seed, oracle_k)
        oracle_edges = build_oracle_edges(
            text_embeddings, dataframe, test_unseen, train_indices, oracle_k
        )
        oracle_data = replace_unseen_edges(data, oracle_edges, test_unseen)
        oracle_loader = _neighbor_loader(
            oracle_data, all_mask, BATCH_SIZE, NUM_NEIGHBORS, shuffle=False
        )
        oracle_embeddings = extract_incident_embeddings(
            model, oracle_data, oracle_loader, device=device
        ).cpu().numpy()
        oracle_metrics = {}
        for group_name, group_indices in [
            ("seen", test_seen),
            ("unseen", test_unseen),
            ("all", test_indices),
        ]:
            oracle_metrics[group_name] = _evaluate_group(
                oracle_embeddings, dataframe, train_indices, group_indices
            )
        oracle_results[oracle_k] = oracle_metrics
    return {
        "seed": seed,
        "training_history": history,
        "control_metrics": control_metrics,
        "oracle_metrics": oracle_results,
    }


def main() -> None:
    """Run oracle edge ceiling experiment and save results."""
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
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--oracle-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 42, 123])
    args = parser.parse_args()
    logger = setup_logging("oracle_edge_ceiling")
    logger.warning(
        "ORACLE MODE: This script uses relevance labels to select edges at inference time. "
        "Results are NOT from a deployable method and represent an upper bound only."
    )
    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, val_indices, test_indices = _split_positions(len(dataframe))
    text_embeddings = None
    if args.features in {"text", "concat"}:
        text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_test_mask = test_cis.notna() & test_cis.isin(train_cis)
    test_seen = test_indices[seen_test_mask.to_numpy()]
    test_unseen = test_indices[~seen_test_mask.to_numpy()]
    logger.info(
        "Dataset: train=%d, val=%d, test=%d (seen=%d, unseen=%d)",
        len(train_indices), len(val_indices), len(test_indices),
        len(test_seen), len(test_unseen),
    )
    logger.info("Running %d seeds with oracle_k=%s", len(args.seeds), args.oracle_k)
    seed_results = []
    failures = []
    for seed in args.seeds:
        try:
            result = run_single_seed(
                seed, args, dataframe, text_embeddings,
                train_indices, val_indices, test_indices,
                test_seen, test_unseen, logger,
            )
            seed_results.append(result)
        except Exception as error:
            logger.exception("Seed %d failed", seed)
            failures.append({"seed": seed, "error": str(error)})
    aggregates = {}
    if seed_results:
        for oracle_k in args.oracle_k:
            aggregates[oracle_k] = {}
            for group in ["seen", "unseen", "all"]:
                control_ndcg1 = [r["control_metrics"][group]["ndcg@1"] for r in seed_results]
                control_map = [r["control_metrics"][group]["map"] for r in seed_results]
                oracle_ndcg1 = [
                    r["oracle_metrics"][oracle_k][group]["ndcg@1"] for r in seed_results
                ]
                oracle_map = [
                    r["oracle_metrics"][oracle_k][group]["map"] for r in seed_results
                ]
                aggregates[oracle_k][group] = {
                    "control_ndcg@1_mean": statistics.mean(control_ndcg1),
                    "control_ndcg@1_std": statistics.pstdev(control_ndcg1),
                    "control_map_mean": statistics.mean(control_map),
                    "control_map_std": statistics.pstdev(control_map),
                    "oracle_ndcg@1_mean": statistics.mean(oracle_ndcg1),
                    "oracle_ndcg@1_std": statistics.pstdev(oracle_ndcg1),
                    "oracle_map_mean": statistics.mean(oracle_map),
                    "oracle_map_std": statistics.pstdev(oracle_map),
                    "delta_ndcg@1": statistics.mean(oracle_ndcg1) - statistics.mean(control_ndcg1),
                    "delta_map": statistics.mean(oracle_map) - statistics.mean(control_map),
                }
    logger.info("Oracle edge ceiling results:")
    logger.info("oracle_k  group    control_ndcg@1      oracle_ndcg@1       control_map         oracle_map          delta_ndcg@1  delta_map")
    for oracle_k in sorted(aggregates.keys()):
        for group in ["seen", "unseen", "all"]:
            stats = aggregates[oracle_k][group]
            logger.info(
                "%-9d %-8s %.3f ± %.3f         %.3f ± %.3f         %.3f ± %.3f         %.3f ± %.3f         %+.3f         %+.3f",
                oracle_k, group,
                stats["control_ndcg@1_mean"], stats["control_ndcg@1_std"],
                stats["oracle_ndcg@1_mean"], stats["oracle_ndcg@1_std"],
                stats["control_map_mean"], stats["control_map_std"],
                stats["oracle_map_mean"], stats["oracle_map_std"],
                stats["delta_ndcg@1"], stats["delta_map"],
            )
    output = {
        "oracle": True,
        "config": vars(args),
        "seed_results": seed_results,
        "aggregates": aggregates,
        "failures": failures,
    }
    output_path = Path("results/diagnostics/oracle_edge_ceiling.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
