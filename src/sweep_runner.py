"""Shared native-wandb sweep machinery for SN incident retrieval."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import statistics
import subprocess
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

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
from src.models import HeteroIncidentClassifier, extract_incident_embeddings, train_minibatch
from src.relevance import compute_relevance_matrix, compute_relevance_matrix_e


BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
WANDB_PROJECT = "gnn-incident-retrieval"
RESULTS_ROOT = "results/sweeps"
_CONTEXT_CACHE: dict = {}
_LOGGER = setup_logging("sweep_runner")


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


def _evaluate_group(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    relevance_definition: str = "D",
) -> dict[str, float]:
    """Evaluate one test CI-visibility group against all train candidates."""
    if relevance_definition == "D":
        relevance = compute_relevance_matrix(
            dataframe, train_indices, test_indices, definition="D"
        )
    elif relevance_definition == "E":
        relevance = compute_relevance_matrix_e(dataframe, train_indices, test_indices)
    else:
        raise ValueError("relevance_definition must be 'D' or 'E'")
    query = embeddings[test_indices]
    candidates = embeddings[train_indices]
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    similarities = np.divide(
        query, query_norm, out=np.zeros_like(query, dtype=np.float32), where=query_norm != 0
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


def _evaluate_retrieval_groups(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seen_test_indices: np.ndarray,
    unseen_test_indices: np.ndarray,
    relevance_definition: str,
) -> dict[str, dict[str, float]]:
    """Evaluate retrieval metrics for seen, unseen, and all test groups."""
    results: dict[str, dict[str, float]] = {}
    for group_name, group_indices in (
        ("seen", seen_test_indices),
        ("unseen", unseen_test_indices),
        ("all", test_indices),
    ):
        if len(group_indices) == 0:
            continue
        results[group_name] = _evaluate_group(
            embeddings,
            dataframe,
            train_indices,
            group_indices,
            relevance_definition=relevance_definition,
        )
    return results


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


def _build_node_counts(data: HeteroData) -> dict[str, int]:
    """Extract positive non-incident node counts from a graph."""
    return {
        node_type: data[node_type].num_nodes
        for node_type in data.node_types
        if node_type != "incident" and data[node_type].num_nodes > 0
    }


def build_graph_for_features(
    features: str,
    dataframe: pd.DataFrame,
    text_embeddings: np.ndarray | None,
    knn_edges: int = 0,
    knn_scope: str = "all",
    train_indices: np.ndarray | None = None,
    unseen_test_indices: np.ndarray | None = None,
    add_assignment_group: bool = False,
) -> HeteroData:
    """Build incident graph for one feature mode and optional text kNN edges."""
    if knn_edges < 0:
        raise ValueError("knn_edges must be non-negative")
    if knn_scope not in {"all", "unseen_only"}:
        raise ValueError("knn_scope must be all or unseen_only")
    if features == "categorical" and knn_edges == 0:
        return build_incident_graph_no_target(
            dataframe, add_assignment_group=add_assignment_group
        )
    if text_embeddings is None:
        raise ValueError("text embeddings required for text, concat, and kNN features")
    if features == "text":
        data = build_incident_graph_no_target(
            dataframe,
            incident_features=text_embeddings,
            add_assignment_group=add_assignment_group,
        )
    elif features == "concat":
        data = build_incident_graph_no_target(
            dataframe, add_assignment_group=add_assignment_group
        )
        base_features = data["incident"].x
        if base_features.size(0) != text_embeddings.shape[0]:
            raise ValueError("base and text feature row counts do not match")
        data["incident"].x = torch.cat(
            [base_features, torch.as_tensor(text_embeddings, dtype=torch.float32)], dim=1
        )
    else:
        data = build_incident_graph_no_target(
            dataframe, add_assignment_group=add_assignment_group
        )
    if knn_edges > 0:
        if train_indices is None:
            raise ValueError("train_indices required for kNN edges")
        if knn_scope == "unseen_only" and unseen_test_indices is None:
            raise ValueError("unseen_test_indices required for unseen_only kNN edges")
        sources = (
            np.arange(len(dataframe), dtype=np.int64)
            if knn_scope == "all"
            else np.asarray(unseen_test_indices, dtype=np.int64)
        )
        from src.knn_edges import add_knn_relation, build_text_knn_edges

        edge_index, stats = build_text_knn_edges(
            text_embeddings, sources, train_indices, knn_edges
        )
        add_knn_relation(data, edge_index)
        _LOGGER.info(
            "Added text kNN edges: count=%d mean_similarity=%.4f edge_types=%s",
            stats["num_edges"], stats["mean_similarity"], data.metadata()[1],
        )
    return data


def _git_commit() -> str | None:
    """Return current git commit hash, or None when unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: object) -> None:
    """Write JSON value to path with stable readable formatting."""
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)


def _metric(results: Mapping[str, Mapping[str, float]], group: str, name: str) -> float:
    """Return group metric or zero when group was empty."""
    return float(results.get(group, {}).get(name, 0.0))


def get_shared_context(
    subtype: str,
    needed_features: Iterable[str],
    knn_edges: int = 0,
    knn_scope: str = "all",
    add_assignment_group: bool = False,
) -> dict:
    """Prepare and cache dataframe, splits, masks, and requested feature graphs."""
    requested = list(dict.fromkeys(needed_features))
    context = _CONTEXT_CACHE.get(subtype)
    if context is None:
        dataframe, _, graph_row_indices = _prepare_dataframe(subtype)
        train_indices, val_indices, test_indices = _split_positions(len(dataframe))
        masks = make_split_masks(dataframe)
        train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
        test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
        seen_mask = test_cis.notna() & test_cis.isin(train_cis)
        seen_test_indices = test_indices[seen_mask.to_numpy()]
        unseen_test_indices = test_indices[~seen_mask.to_numpy()]
        context = {
            "dataframe": dataframe,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "seen_test_indices": seen_test_indices,
            "unseen_test_indices": unseen_test_indices,
            "masks": masks,
            "graphs": {},
            "num_classes": int(dataframe["Closure Code"].nunique()),
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "graph_row_indices": graph_row_indices,
            "text_embeddings": None,
        }
        _CONTEXT_CACHE[subtype] = context
        _LOGGER.info("Prepared shared context for subtype=%s", subtype)
    else:
        _LOGGER.info("Reusing shared context for subtype=%s", subtype)

    text_needed = knn_edges > 0 or any(feature in {"text", "concat"} for feature in requested)
    if text_needed and context["text_embeddings"] is None:
        context["text_embeddings"] = np.load("results/sn_text_embeddings.npy")[
            context["graph_row_indices"]
        ]
    for features in requested:
        graph_key = (features, knn_edges, knn_scope, add_assignment_group)
        if graph_key not in context["graphs"]:
            graph = build_graph_for_features(
                features,
                context["dataframe"],
                context["text_embeddings"],
                knn_edges=knn_edges,
                knn_scope=knn_scope,
                train_indices=context["train_indices"],
                unseen_test_indices=context["unseen_test_indices"],
                add_assignment_group=add_assignment_group,
            )
            context["graphs"][graph_key] = graph
            _LOGGER.info(
                "Built %s graph (knn_edges=%d, knn_scope=%s) with input_dim=%d",
                features,
                knn_edges,
                knn_scope,
                graph["incident"].x.size(1),
            )
    return context


def build_run_name(config: Mapping[str, object]) -> str:
    """Build deterministic filesystem-safe run name from config."""
    def clean(value: object) -> str:
        """Format one run-name value for filesystem use."""
        return str(value).replace(".", "p")

    weight_decay = float(config.get("weight_decay", 1e-4))
    weight_decay_name = f"_wd{clean(weight_decay)}" if weight_decay != 1e-4 else ""
    run_name = (
        f"{config['features']}_embed{config['entity_embed_dim']}"
        f"_drop{clean(config['ci_dropout_rate'])}"
        f"_{config['ci_dropout_mode']}_lr{clean(config['lr'])}"
        f"{weight_decay_name}_h{config.get('hidden_dim', 128)}"
        f"_l2_do{clean(config.get('dropout', 0.3))}"
        f"_s{config['seed']}_knn{config.get('knn_edges', 0)}"
        f"{'a' if config.get('knn_scope', 'all') == 'all' else 'u'}"
    )
    triplet_weight = float(config.get("triplet_weight", 0.0))
    if triplet_weight > 0:
        run_name += (
            f"_tw{clean(triplet_weight)}_tm{clean(config.get('triplet_margin', 0.2))}"
            f"_cl{config.get('contrastive_loss', 'triplet')}"
            f"_t{clean(config.get('temperature', 0.07))}"
        )
    if bool(config.get("add_assignment_group", False)):
        run_name += "_ag"
    assert "/" not in run_name and "\\" not in run_name
    assert not any(character.isspace() for character in run_name)
    return run_name


def run_one(
    config: Mapping[str, object],
    sweep_name: str,
    sweep_notes: str,
    epochs: int,
    patience: int,
    subtype: str,
    run_name: str | None = None,
    knn_edges: int = 0,
    knn_scope: str = "all",
) -> dict:
    """Train, evaluate, and record one configuration in an existing wandb run."""
    _set_seed(int(config["seed"]))
    hidden_dim = int(wandb.config.get("hidden_dim", config.get("hidden_dim", 128)))
    dropout = float(wandb.config.get("dropout", config.get("dropout", 0.3)))
    weight_decay = float(
        wandb.config.get("weight_decay", config.get("weight_decay", 1e-4))
    )
    triplet_weight = float(
        wandb.config.get("triplet_weight", config.get("triplet_weight", 0.0))
    )
    triplet_margin = float(
        wandb.config.get("triplet_margin", config.get("triplet_margin", 0.2))
    )
    contrastive_loss = str(
        wandb.config.get("contrastive_loss", config.get("contrastive_loss", "triplet"))
    )
    temperature = float(
        wandb.config.get("temperature", config.get("temperature", 0.07))
    )
    add_assignment_group = bool(
        wandb.config.get("add_assignment_group", config.get("add_assignment_group", False))
    )
    relevance_definition = str(
        wandb.config.get("relevance_def", config.get("relevance_def", "D"))
    )
    eval_every_epoch = bool(
        wandb.config.get("eval_every_epoch", config.get("eval_every_epoch", False))
    )
    if relevance_definition not in {"D", "E"}:
        raise ValueError("relevance_def must be 'D' or 'E'")
    if contrastive_loss not in {"triplet", "infonce"}:
        raise ValueError("contrastive_loss must be triplet or infonce")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    run_config = {
        **config,
        "contrastive_loss": contrastive_loss,
        "temperature": temperature,
        "relevance_def": relevance_definition,
        "eval_every_epoch": eval_every_epoch,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "triplet_weight": triplet_weight,
        "triplet_margin": triplet_margin,
        "add_assignment_group": add_assignment_group,
    }
    run_name = run_name or build_run_name(run_config)
    knn_edges = int(config.get("knn_edges", knn_edges))
    knn_scope = str(config.get("knn_scope", knn_scope))
    context = get_shared_context(
        subtype,
        [str(config["features"])],
        knn_edges,
        knn_scope,
        add_assignment_group,
    )
    dataframe = context["dataframe"]
    data = context["graphs"][
        (config["features"], knn_edges, knn_scope, add_assignment_group)
    ]
    train_indices = context["train_indices"]
    test_indices = context["test_indices"]
    input_dim = data["incident"].x.size(1)
    run_config.update({"knn_edges": knn_edges, "knn_scope": knn_scope})
    wandb.config.update(
        {
            **run_config,
            "input_dim": input_dim,
            "num_cis": data["ci"].num_nodes,
            "num_incidents": len(dataframe),
            "num_train": len(train_indices),
            "num_test": len(test_indices),
            "num_test_seen": len(context["seen_test_indices"]),
            "num_test_unseen": len(context["unseen_test_indices"]),
            "batch_size": BATCH_SIZE,
            "num_neighbors": NUM_NEIGHBORS,
            "epochs_max": epochs,
            "patience": patience,
            "subtype": subtype,
            "sweep_name": sweep_name,
            "git_commit": _git_commit(),
        },
        allow_val_change=True,
    )
    model = HeteroIncidentClassifier(
        data.metadata(),
        node_counts=_build_node_counts(data),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=context["num_classes"],
        num_layers=int(config["num_layers"]),
        dropout=dropout,
        ci_dropout_rate=float(config["ci_dropout_rate"]),
        ci_dropout_mode=str(config["ci_dropout_mode"]),
        entity_embed_dim=int(config["entity_embed_dim"]),
    )
    run_started = time.perf_counter()
    retrieval_history: list[dict[str, float | int]] = []
    first_retrieval_eval_sec: float | None = None
    full_loader = (
        NeighborLoader(
            data,
            num_neighbors=NUM_NEIGHBORS,
            input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        if eval_every_epoch
        else None
    )
    original_wandb_log = wandb.log

    def log_with_epoch_retrieval(
        payload: dict[str, object] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Add retrieval metrics to each training epoch W&B log."""
        nonlocal first_retrieval_eval_sec
        if (
            eval_every_epoch
            and payload is not None
            and "epoch" in payload
            and "train_loss" in payload
            and "val_loss" in payload
        ):
            started = time.perf_counter()
            embeddings = extract_incident_embeddings(
                model, data, full_loader, str(context["device"])
            ).numpy()
            epoch_results = _evaluate_retrieval_groups(
                embeddings,
                dataframe,
                train_indices,
                test_indices,
                context["seen_test_indices"],
                context["unseen_test_indices"],
                relevance_definition,
            )
            elapsed = time.perf_counter() - started
            if first_retrieval_eval_sec is None:
                first_retrieval_eval_sec = elapsed
                _LOGGER.info(
                    "First per-epoch retrieval evaluation took %.3fs", elapsed
                )
            epoch = int(payload["epoch"])
            retrieval_entry: dict[str, float | int] = {"epoch": epoch}
            for group_name in ("seen", "unseen", "all"):
                group_metrics = epoch_results.get(group_name, {})
                retrieval_entry[f"{group_name}_ndcg1"] = group_metrics.get(
                    "ndcg@1", 0.0
                )
                if group_name in {"seen", "unseen"}:
                    retrieval_entry[f"{group_name}_mrr"] = group_metrics.get(
                        "mrr", 0.0
                    )
                    retrieval_entry[f"{group_name}_map"] = group_metrics.get(
                        "map", 0.0
                    )
            retrieval_history.append(retrieval_entry)
            payload = {**payload, **retrieval_entry}
        original_wandb_log(payload, *args, **kwargs)

    if eval_every_epoch:
        wandb.log = log_with_epoch_retrieval
    try:
        model, history = train_minibatch(
            model,
            data,
            context["masks"]["train_mask"],
            context["masks"]["val_mask"],
            epochs=epochs,
            lr=float(config["lr"]),
            weight_decay=weight_decay,
            batch_size=BATCH_SIZE,
            num_neighbors=NUM_NEIGHBORS,
            device=str(context["device"]),
            patience=patience,
            triplet_weight=triplet_weight,
            triplet_margin=triplet_margin,
            contrastive_loss=contrastive_loss,
            temperature=temperature,
        )
    finally:
        if eval_every_epoch:
            wandb.log = original_wandb_log
    if full_loader is None:
        full_loader = NeighborLoader(
            data,
            num_neighbors=NUM_NEIGHBORS,
            input_nodes=("incident", torch.ones(len(dataframe), dtype=torch.bool)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    embeddings = extract_incident_embeddings(
        model, data, full_loader, str(context["device"])
    ).numpy()
    results = _evaluate_retrieval_groups(
        embeddings,
        dataframe,
        train_indices,
        test_indices,
        context["seen_test_indices"],
        context["unseen_test_indices"],
        relevance_definition,
    )
    for group_name, metrics in results.items():
        for metric_name, metric_value in metrics.items():
            wandb.log({f"{group_name}/{metric_name}": metric_value})
    best_val_loss = min(history["val_loss"])
    summary_values = {
        "seen_ndcg1": _metric(results, "seen", "ndcg@1"),
        "unseen_ndcg1": _metric(results, "unseen", "ndcg@1"),
        "all_ndcg1": _metric(results, "all", "ndcg@1"),
        "unseen_map": _metric(results, "unseen", "map"),
        "unseen_mrr": _metric(results, "unseen", "mrr"),
        "seen_map": _metric(results, "seen", "map"),
        "seen_mrr": _metric(results, "seen", "mrr"),
        "best_epoch": history["best_epoch"],
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "stopped_early": history["stopped_early"],
        "seen_unseen_gap": _metric(results, "seen", "ndcg@1") - _metric(results, "unseen", "ndcg@1"),
        "first_retrieval_eval_sec": first_retrieval_eval_sec,
        "run_duration_sec": time.perf_counter() - run_started,
    }
    for key, value in summary_values.items():
        wandb.summary[key] = value
    run = wandb.run
    record = {
        "config": run_config,
        "input_dim": input_dim,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "best_val_loss": best_val_loss,
        "stopped_early": history["stopped_early"],
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "val_acc_history": history["val_acc"],
        "retrieval_history": retrieval_history,
        "results": results,
        "first_retrieval_eval_sec": first_retrieval_eval_sec,
        "run_duration_sec": summary_values["run_duration_sec"],
        "wandb_run_id": getattr(run, "id", None),
        "wandb_run_url": getattr(run, "url", None),
    }
    output_path = Path(RESULTS_ROOT) / sweep_name / "runs" / f"{run_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, record)
    _LOGGER.info(
        "Completed %s | best_epoch=%d | seen=%.3f unseen=%.3f all=%.3f",
        run_name, history["best_epoch"], summary_values["seen_ndcg1"],
        summary_values["unseen_ndcg1"], summary_values["all_ndcg1"],
    )
    return record


_AGGREGATE_METRICS = (
    "seen_ndcg1", "unseen_ndcg1", "all_ndcg1", "seen_map", "unseen_map",
    "all_map", "seen_mrr", "unseen_mrr", "all_mrr", "best_epoch",
    "best_val_loss", "seen_unseen_gap",
)


def aggregate_over_seeds(run_records: list[dict]) -> list[dict]:
    """Group run records by configuration excluding seed and aggregate metrics."""
    groups: dict[tuple[tuple[str, object], ...], list[dict]] = {}
    for record in run_records:
        config = record["config"]
        key = tuple(sorted((name, value) for name, value in config.items() if name != "seed"))
        groups.setdefault(key, []).append(record)
    aggregated = []
    for key, records in groups.items():
        config = dict(key)
        seeds = [record["config"]["seed"] for record in records]
        run_name = record_name = records[0].get("run_name", build_run_name(records[0]["config"]))
        config_label = re.sub(r"_s[^_]+$", "", run_name)
        row = {**config, "config_label": config_label, "seeds": seeds, "n_seeds": len(seeds)}
        for metric in _AGGREGATE_METRICS:
            values = [
                float(record.get(metric, record.get("results", {}).get(metric, 0.0)))
                if metric in record else _record_metric(record, metric)
                for record in records
            ]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        row["run_name"] = record_name
        aggregated.append(row)
    return aggregated


def _record_metric(record: dict, metric: str) -> float:
    """Extract summary metric from one run JSON record."""
    if metric == "best_epoch":
        return float(record["best_epoch"])
    if metric == "best_val_loss":
        return float(record["best_val_loss"])
    results = record.get("results", {})
    if metric == "seen_unseen_gap":
        return _metric(results, "seen", "ndcg@1") - _metric(results, "unseen", "ndcg@1")
    group, name = metric.split("_", 1)
    return _metric(results, group, "ndcg@1" if name == "ndcg1" else name)


def write_sweep_summaries(
    sweep_name: str,
    sweep_notes: str,
    total_runs: int,
    failures: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Read sweep run JSONs and write per-run and per-config summaries."""
    output_dir = Path(RESULTS_ROOT) / sweep_name
    records = []
    for path in sorted((output_dir / "runs").glob("*.json")):
        with path.open(encoding="utf-8") as input_file:
            record = json.load(input_file)
        record["run_name"] = path.stem
        records.append(record)
    csv_fields = [
        "run_name", "features", "entity_embed_dim", "ci_dropout_rate", "ci_dropout_mode",
        "lr", "weight_decay", "hidden_dim", "num_layers", "dropout", "triplet_weight",
        "triplet_margin", "seed", "knn_edges", "knn_scope", "input_dim",
        "best_epoch", "epochs_trained", "best_val_loss", "seen_ndcg1", "unseen_ndcg1",
        "all_ndcg1", "seen_map", "unseen_map", "all_map", "seen_mrr", "unseen_mrr", "all_mrr",
        "seen_unseen_gap",
    ]
    rows = []
    for record in records:
        config, results = record["config"], record["results"]
        row = {
            "run_name": record["run_name"],
            **{field: config[field] for field in csv_fields[1:15]},
            "input_dim": record["input_dim"], "best_epoch": record["best_epoch"],
            "epochs_trained": record["epochs_trained"], "best_val_loss": record["best_val_loss"],
            "seen_ndcg1": _metric(results, "seen", "ndcg@1"),
            "unseen_ndcg1": _metric(results, "unseen", "ndcg@1"),
            "all_ndcg1": _metric(results, "all", "ndcg@1"),
            "seen_map": _metric(results, "seen", "map"), "unseen_map": _metric(results, "unseen", "map"),
            "all_map": _metric(results, "all", "map"), "seen_mrr": _metric(results, "seen", "mrr"),
            "unseen_mrr": _metric(results, "unseen", "mrr"), "all_mrr": _metric(results, "all", "mrr"),
            "seen_unseen_gap": _metric(results, "seen", "ndcg@1") - _metric(results, "unseen", "ndcg@1"),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["unseen_ndcg1"], reverse=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        f"# {sweep_name}", "", f"Notes: {sweep_notes}", f"Total runs: {total_runs}",
        f"Failed runs: {len(failures or [])}", "", "| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| {row['run_name']} | {row['best_epoch']} | {row['seen_ndcg1']:.6f} | {row['unseen_ndcg1']:.6f} | {row['all_ndcg1']:.6f} | {row['seen_unseen_gap']:.6f} |"
        for row in rows
    )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    config_rows = aggregate_over_seeds(records)
    config_rows.sort(key=lambda row: row["unseen_ndcg1_mean"], reverse=True)
    config_fields = [
        "config_label", "features", "entity_embed_dim", "ci_dropout_rate", "ci_dropout_mode", "lr",
        "weight_decay", "hidden_dim", "num_layers", "dropout", "triplet_weight",
        "triplet_margin", "knn_edges", "knn_scope",
        "n_seeds", "seeds",
        *[f"{metric}_{suffix}" for metric in _AGGREGATE_METRICS for suffix in ("mean", "std")],
    ]
    with (output_dir / "summary_by_config.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in config_fields} for row in config_rows)
    config_markdown = [
        f"# {sweep_name} by configuration", "", f"Notes: {sweep_notes}", "",
        "| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in config_rows:
        def formatted(metric: str) -> str:
            """Format aggregated metric as mean plus or minus standard deviation."""
            return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
        config_markdown.append(
            f"| {row['config_label']} | {row['n_seeds']} | {formatted('seen_ndcg1')} | {formatted('unseen_ndcg1')} | {formatted('all_ndcg1')} | {row['best_epoch_mean']:.3f} |"
        )
    multi_seed = [row["unseen_ndcg1_std"] for row in config_rows if row["n_seeds"] > 1]
    if multi_seed:
        config_markdown.extend(["", f"Observed noise floor (largest unseen_ndcg1 std): {max(multi_seed):.3f}"])
    (output_dir / "summary_by_config.md").write_text("\n".join(config_markdown) + "\n", encoding="utf-8")
    return rows, config_rows
