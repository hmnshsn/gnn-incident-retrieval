"""Train a text-to-CI predictor and evaluate CI text profiling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.ci_text_profiler import (
    aggregate_ci_text_profiles,
    aggregate_unseen_ci_text_profiles,
    predict_unseen_ci_embeddings,
    train_ci_text_predictor,
)
from src.data_loader import temporal_split
from src.data_loader_sn import load_sn_incidents
from src.evaluate_retrieval import evaluate_retrieval_graded
from src.graph_builder import build_incident_graph_no_target, make_split_masks
from src.impute_subcategory import impute_subcategory
from src.logging_config import setup_logging
from src.models import HeteroIncidentClassifier, extract_incident_embeddings, train_minibatch
from src.relevance import compute_relevance_matrix


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
RETRIEVAL_KS = (1, 5, 10, 20)


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
    return np.arange(train_end), np.arange(train_end, val_end), np.arange(val_end, n_rows)


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows, preserving zero rows."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings, dtype=np.float32), where=norms != 0)


def _cosine_similarity(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and candidate embeddings."""
    return _normalize_rows(query_embeddings) @ _normalize_rows(candidate_embeddings).T


def _metrics_for_similarity(similarities: np.ndarray, relevance: object) -> dict[str, float]:
    """Evaluate similarity scores with requested lowercase metric keys."""
    metrics = evaluate_retrieval_graded(similarities, relevance, ks=RETRIEVAL_KS)
    return {key: metrics[source] for key, source in {
        "ndcg@1": "nDCG@1", "ndcg@5": "nDCG@5", "map": "MAP", "mrr": "MRR"
    }.items()}


def _build_node_counts(data: HeteroData) -> dict[str, int]:
    """Extract positive non-incident node counts from a graph."""
    return {node_type: data[node_type].num_nodes for node_type in data.node_types if node_type != "incident" and data[node_type].num_nodes > 0}


def _training_ci_indices(data: HeteroData, train_indices: np.ndarray) -> set[int]:
    """Find CI nodes connected to training incidents."""
    train_mask = np.zeros(data["incident"].num_nodes, dtype=bool)
    train_mask[train_indices] = True
    seen: set[int] = set()
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_type[0] == "incident" and edge_type[2] == "ci":
            incident_indices, ci_indices = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        elif edge_type[0] == "ci" and edge_type[2] == "incident":
            ci_indices, incident_indices = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        else:
            continue
        seen.update(ci_indices[train_mask[incident_indices]].tolist())
    return seen


def _evaluate_group(embeddings: np.ndarray, dataframe: pd.DataFrame, train_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, float]:
    """Evaluate one test CI-visibility group against all train candidates."""
    relevance = compute_relevance_matrix(dataframe, train_indices, test_indices)
    return _metrics_for_similarity(_cosine_similarity(embeddings[test_indices], embeddings[train_indices]), relevance)


def _evaluate_all_groups(embeddings: np.ndarray, dataframe: pd.DataFrame, train_indices: np.ndarray, test_indices: np.ndarray, seen_test_indices: np.ndarray, unseen_test_indices: np.ndarray) -> dict[str, dict[str, float]]:
    """Evaluate seen, unseen, and combined test groups."""
    return {group: _evaluate_group(embeddings, dataframe, train_indices, indices) for group, indices in {
        "seen": seen_test_indices, "unseen": unseen_test_indices, "all": test_indices
    }.items()}


def _format_summary(baseline: dict[str, dict[str, float]], profiled: dict[str, dict[str, float]]) -> str:
    """Format baseline and text-profiled retrieval metrics."""
    lines = ["Group   baseline nDCG@1/MAP/MRR   profiled nDCG@1/MAP/MRR"]
    for group in ("seen", "unseen", "all"):
        base, profile = baseline[group], profiled[group]
        lines.append(f"{group:<7} {base['ndcg@1']:.3f}/{base['map']:.3f}/{base['mrr']:.3f}             {profile['ndcg@1']:.3f}/{profile['map']:.3f}/{profile['mrr']:.3f}")
    return "\n".join(lines)


def main() -> None:
    """Train GNN, profile CI embeddings from text, and save metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-dropout-rate", type=float, default=0.0)
    parser.add_argument("--ci-dropout-mode", choices=["embedding", "edge", "both"], default="embedding")
    parser.add_argument("--features", choices=["tfidf", "text", "concat"], default="tfidf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entity-embed-dim", type=int, default=None)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.ci_dropout_rate <= 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("invalid dropout rate, epochs, or patience")
    logger = setup_logging("sn_exp11_ci_text_profiling")
    _set_seed(args.seed)

    logger.info("Step 1: preparing data")
    dataframe, splits, graph_row_indices = _prepare_dataframe(args.subtype)
    train_df, val_df, test_df = splits
    train_indices, _, test_indices = _split_positions(len(dataframe))
    masks = make_split_masks(dataframe)
    text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    if args.features == "text":
        data = build_incident_graph_no_target(dataframe, incident_features=text_embeddings)
    else:
        data = build_incident_graph_no_target(dataframe)
        if args.features == "concat":
            data["incident"].x = torch.as_tensor(np.concatenate([data["incident"].x.numpy(), text_embeddings], axis=1), dtype=torch.float32)

    logger.info("Step 2: training GNN")
    model = HeteroIncidentClassifier(data.metadata(), _build_node_counts(data), data["incident"].x.size(1), hidden_dim=128, num_classes=int(dataframe["Closure Code"].nunique()), num_layers=2, dropout=0.3, ci_dropout_rate=args.ci_dropout_rate, ci_dropout_mode=args.ci_dropout_mode, entity_embed_dim=args.entity_embed_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)), batch_size=BATCH_SIZE, shuffle=False)
    model, history = train_minibatch(model, data, masks["train_mask"], masks["val_mask"], epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=BATCH_SIZE, num_neighbors=NUM_NEIGHBORS, device=str(device), patience=args.patience)

    logger.info("Step 3: extracting baseline embeddings")
    baseline_embeddings = extract_incident_embeddings(model, data, full_loader, str(device)).numpy()
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]

    logger.info("Step 4: evaluating baseline")
    baseline_results = _evaluate_all_groups(baseline_embeddings, dataframe, train_indices, test_indices, seen_test_indices, unseen_test_indices)
    logger.info("Step 5: building CI text profiles")
    ci_text_profiles, ci_counts = aggregate_ci_text_profiles(text_embeddings, dataframe, train_indices, data)
    logger.info("CI profiles: nonzero=%d, zero=%d", int((ci_counts > 0).sum()), int((ci_counts == 0).sum()))

    logger.info("Step 6: training text-to-CI predictor")
    learned_ci = model.entity_embeddings["ci"].weight.detach().cpu().numpy()
    seen_ci_indices = _training_ci_indices(data, train_indices)
    predictor, predictor_stats = train_ci_text_predictor(ci_text_profiles, learned_ci, seen_ci_indices, args.seed)

    unseen_ci_indices = set(range(data["ci"].num_nodes)) - seen_ci_indices
    logger.info("Step 6.5: building unseen CI text profiles from test incidents")
    unseen_ci_text_profiles, unseen_ci_counts = aggregate_unseen_ci_text_profiles(
        text_embeddings, dataframe, test_indices, data, unseen_ci_indices
    )
    logger.info(
        "Unseen CI test profiles: nonzero=%d, zero=%d",
        int((unseen_ci_counts > 0).sum()),
        int((unseen_ci_counts == 0).sum()),
    )

    logger.info("Step 7: predicting unseen CI embeddings")
    predicted, prediction_stats = predict_unseen_ci_embeddings(
        predictor, unseen_ci_text_profiles, unseen_ci_indices
    )

    logger.info("Step 8: re-inferring with text-profiled CIs")
    hybrid = learned_ci.copy()
    for ci_index, embedding in predicted.items():
        hybrid[ci_index] = embedding
    profiled_embeddings = extract_incident_embeddings(model, data, full_loader, str(device), torch.as_tensor(hybrid, dtype=torch.float32), "inductive").numpy()
    logger.info("Step 9: evaluating text-profiled embeddings")
    profiled_results = _evaluate_all_groups(profiled_embeddings, dataframe, train_indices, test_indices, seen_test_indices, unseen_test_indices)
    logger.info("Step 10: comparison\n%s", _format_summary(baseline_results, profiled_results))

    output_path = Path(
        f"results/exp11_sn_ci_text_profiling/"
        f"ci_text_profiling_{args.features}_dropout{args.ci_dropout_rate}_"
        f"lr{args.lr}_wd{args.weight_decay}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "ci_dropout_rate": args.ci_dropout_rate, "ci_dropout_mode": args.ci_dropout_mode,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "entity_embed_dim": args.entity_embed_dim,
        "features": args.features, "subtype": args.subtype,
        "epochs_trained": len(history["train_loss"]), "best_epoch": history["best_epoch"],
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "stopped_early": history["stopped_early"], "num_test_seen": len(seen_test_indices),
        "num_test_unseen": len(unseen_test_indices), "predictor_stats": predictor_stats,
        "prediction_stats": prediction_stats, "baseline": baseline_results, "profiled": profiled_results,
    }, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
