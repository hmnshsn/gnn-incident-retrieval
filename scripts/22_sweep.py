"""Run named, configurable sweeps for SN incident retrieval."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

import os

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
RESULTS_ROOT = "results/sweeps"


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


def build_graph_for_features(
    features: str,
    dataframe: pd.DataFrame,
    text_embeddings: np.ndarray | None,
) -> HeteroData:
    """Build the incident graph for a given feature mode.

    Args:
        features: One of ``categorical``, ``text``, or ``concat``.
        dataframe: Prepared incident dataframe.
        text_embeddings: Cached description embeddings aligned to dataframe rows,
            or None when features is ``categorical``.

    Returns:
        HeteroData graph with incident node features set for this mode.
    """
    if features == "categorical":
        return build_incident_graph_no_target(dataframe)
    if text_embeddings is None:
        raise ValueError("text embeddings required for text and concat features")
    if features == "text":
        return build_incident_graph_no_target(
            dataframe,
            incident_features=text_embeddings,
        )
    data = build_incident_graph_no_target(dataframe)
    base_features = data["incident"].x
    if base_features.size(0) != text_embeddings.shape[0]:
        raise ValueError("base and text feature row counts do not match")
    data["incident"].x = torch.cat(
        [base_features, torch.as_tensor(text_embeddings, dtype=torch.float32)],
        dim=1,
    )
    return data


def build_run_name(config: Mapping[str, object]) -> str:
    """Build a deterministic, filesystem-safe run name from a config dict.

    Only includes axes that vary across the sweep is NOT required -- always
    include the full set so names are unambiguous across sweeps.

    Args:
        config: Mapping of hyperparameter name to value.

    Returns:
        String run name safe for use as a filename.
    """
    def clean(value: object) -> str:
        """Format one run-name value for filesystem use."""
        return str(value).replace(".", "p")

    run_name = (
        f"{config['features']}_embed{config['entity_embed_dim']}"
        f"_drop{clean(config['ci_dropout_rate'])}"
        f"_{config['ci_dropout_mode']}_lr{clean(config['lr'])}"
        f"_wd{clean(config['weight_decay'])}_h{config['hidden_dim']}"
        f"_l{config['num_layers']}_do{clean(config['dropout'])}"
        f"_s{config['seed']}"
    )
    assert "/" not in run_name and "\\" not in run_name
    assert not any(character.isspace() for character in run_name)
    return run_name


def _git_commit() -> str | None:
    """Return current git commit hash, or None when unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
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


def _write_summary(
    output_dir: Path,
    sweep_name: str,
    sweep_notes: str,
    total_runs: int,
    failures: list[dict[str, str]],
    run_names: list[str],
) -> list[dict[str, object]]:
    """Read run JSONs and write CSV and Markdown sweep summaries."""
    records: list[dict[str, object]] = []
    for run_name in run_names:
        path = output_dir / "runs" / f"{run_name}.json"
        if path.exists():
            with path.open(encoding="utf-8") as input_file:
                record = json.load(input_file)
            record["run_name"] = run_name
            records.append(record)

    csv_fields = [
        "run_name", "features", "entity_embed_dim", "ci_dropout_rate",
        "ci_dropout_mode", "lr", "weight_decay", "hidden_dim", "num_layers",
        "dropout", "seed", "input_dim", "best_epoch", "epochs_trained",
        "best_val_loss", "seen_ndcg1", "unseen_ndcg1", "all_ndcg1", "seen_map",
        "unseen_map", "all_map", "seen_mrr", "unseen_mrr", "all_mrr",
        "seen_unseen_gap",
    ]
    rows: list[dict[str, object]] = []
    for record in records:
        config = record["config"]
        results = record["results"]
        row = {
            "run_name": record["run_name"],
            **{field: config[field] for field in csv_fields[1:11]},
            "input_dim": record["input_dim"],
            "best_epoch": record["best_epoch"],
            "epochs_trained": record["epochs_trained"],
            "best_val_loss": record["best_val_loss"],
            "seen_ndcg1": _metric(results, "seen", "ndcg@1"),
            "unseen_ndcg1": _metric(results, "unseen", "ndcg@1"),
            "all_ndcg1": _metric(results, "all", "ndcg@1"),
            "seen_map": _metric(results, "seen", "map"),
            "unseen_map": _metric(results, "unseen", "map"),
            "all_map": _metric(results, "all", "map"),
            "seen_mrr": _metric(results, "seen", "mrr"),
            "unseen_mrr": _metric(results, "unseen", "mrr"),
            "all_mrr": _metric(results, "all", "mrr"),
            "seen_unseen_gap": _metric(results, "seen", "ndcg@1")
            - _metric(results, "unseen", "ndcg@1"),
        }
        rows.append(row)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(rows, key=lambda row: row["unseen_ndcg1"], reverse=True)
    markdown_lines = [
        f"# {sweep_name}",
        "",
        f"Notes: {sweep_notes}",
        f"Total runs: {total_runs}",
        f"Failed runs: {len(failures)}",
        "",
        "| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        markdown_lines.append(
            f"| {row['run_name']} | {row['best_epoch']} | "
            f"{row['seen_ndcg1']:.6f} | {row['unseen_ndcg1']:.6f} | "
            f"{row['all_ndcg1']:.6f} | {row['seen_unseen_gap']:.6f} |"
        )
    (output_dir / "summary.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    return sorted_rows


def _print_dry_run(
    args: argparse.Namespace,
    axes: Mapping[str, list[object]],
    run_names: list[str],
    output_dir: Path,
) -> None:
    """Print resolved sweep configuration without loading data."""
    print(f"sweep_name: {args.sweep_name}")
    print(f"sweep_notes: {args.sweep_notes}")
    print(f"output_dir: {output_dir}")
    for axis, values in axes.items():
        print(f"{axis}: {values}")
    print(f"total_runs: {len(run_names)}")
    print("run_names:")
    for run_name in run_names:
        print(run_name)


def main() -> None:
    """Parse arguments and run one named hyperparameter sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--sweep-notes", default="")
    parser.add_argument("--entity-embed-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--ci-dropout-rates", nargs="+", type=float, default=[0.5])
    parser.add_argument("--lrs", nargs="+", type=float, default=[5e-4])
    parser.add_argument(
        "--features", nargs="+", choices=["categorical", "text", "concat"], default=["concat"]
    )
    parser.add_argument(
        "--ci-dropout-modes", nargs="+", choices=["embedding", "edge", "both"], default=["embedding"]
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--num-layers-list", nargs="+", type=int, default=[2])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    args = parser.parse_args()

    axes: dict[str, list[object]] = {
        "features": args.features,
        "entity_embed_dim": args.entity_embed_dims,
        "ci_dropout_rate": args.ci_dropout_rates,
        "ci_dropout_mode": args.ci_dropout_modes,
        "lr": args.lrs,
        "weight_decay": args.weight_decays,
        "hidden_dim": args.hidden_dims,
        "num_layers": args.num_layers_list,
        "dropout": args.dropouts,
        "seed": args.seeds,
    }
    grid = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    run_names = [build_run_name(config) for config in grid]
    output_dir = Path(RESULTS_ROOT) / args.sweep_name
    if args.dry_run:
        _print_dry_run(args, axes, run_names, output_dir)
        return

    logger = setup_logging(f"sweep_{args.sweep_name}")
    timestamp_start = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sweep_name": args.sweep_name,
        "sweep_notes": args.sweep_notes,
        "timestamp_start": timestamp_start,
        "git_commit": git_commit,
        "axes": axes,
        "epochs": args.epochs,
        "patience": args.patience,
        "subtype": args.subtype,
        "total_runs": len(grid),
        "run_names": run_names,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        logger.warning("Existing manifest found; preserving it")
        manifest_path = output_dir / f"manifest_{datetime.now():%Y%m%d_%H%M%S}.json"
    _write_json(manifest_path, manifest)

    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    masks = make_split_masks(dataframe)
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    text_embeddings = None
    if any(config["features"] in {"text", "concat"} for config in grid):
        text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    graph_cache: dict[str, HeteroData] = {}
    for features in dict.fromkeys(axes["features"]):
        graph_cache[features] = build_graph_for_features(
            features, dataframe, text_embeddings
        )
        logger.info(
            "Built %s graph with input_dim=%d",
            features,
            graph_cache[features]["incident"].x.size(1),
        )

    failures: list[dict[str, str]] = []
    for run_index, (run_name, config) in enumerate(zip(run_names, grid), start=1):
        output_path = runs_dir / f"{run_name}.json"
        if args.skip_existing and output_path.exists():
            logger.info("Skipping existing run %d/%d: %s", run_index, len(grid), run_name)
            continue
        run = None
        run_started = time.perf_counter()
        try:
            _set_seed(config["seed"])
            data = graph_cache[config["features"]]
            input_dim = data["incident"].x.size(1)
            node_counts = _build_node_counts(data)
            model = HeteroIncidentClassifier(
                data.metadata(),
                node_counts=node_counts,
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"],
                num_classes=int(dataframe["Closure Code"].nunique()),
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                ci_dropout_rate=config["ci_dropout_rate"],
                ci_dropout_mode=config["ci_dropout_mode"],
                entity_embed_dim=config["entity_embed_dim"],
            )
            run_config = dict(config)
            wandb_config = {
                **run_config,
                "input_dim": input_dim,
                "num_cis": data["ci"].num_nodes,
                "num_incidents": len(dataframe),
                "num_train": len(train_indices),
                "num_test": len(test_indices),
                "num_test_seen": len(seen_test_indices),
                "num_test_unseen": len(unseen_test_indices),
                "batch_size": BATCH_SIZE,
                "num_neighbors": NUM_NEIGHBORS,
                "epochs_max": args.epochs,
                "patience": args.patience,
                "subtype": args.subtype,
                "sweep_name": args.sweep_name,
                "git_commit": git_commit,
            }
            run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                group=args.sweep_name,
                job_type="sweep_run",
                notes=args.sweep_notes,
                tags=[args.sweep_name] + args.wandb_tags,
                mode=args.wandb_mode,
                config=wandb_config,
                reinit="finish_previous",
            )
            model, history = train_minibatch(
                model,
                data,
                masks["train_mask"],
                masks["val_mask"],
                epochs=args.epochs,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                batch_size=BATCH_SIZE,
                num_neighbors=NUM_NEIGHBORS,
                device=str(device),
                patience=args.patience,
            )
            train_loss_history = history["train_loss"]
            val_loss_history = history["val_loss"]
            val_acc_history = history["val_acc"]
            for epoch, (train_loss, val_loss, val_acc) in enumerate(
                zip(train_loss_history, val_loss_history, val_acc_history), start=1
            ):
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )
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
            best_val_loss = min(val_loss_history)
            run_duration_sec = time.perf_counter() - run_started
            seen_ndcg1 = _metric(results, "seen", "ndcg@1")
            unseen_ndcg1 = _metric(results, "unseen", "ndcg@1")
            all_ndcg1 = _metric(results, "all", "ndcg@1")
            seen_unseen_gap = seen_ndcg1 - unseen_ndcg1
            summary_values = {
                "seen_ndcg1": seen_ndcg1,
                "unseen_ndcg1": unseen_ndcg1,
                "all_ndcg1": all_ndcg1,
                "unseen_map": _metric(results, "unseen", "map"),
                "unseen_mrr": _metric(results, "unseen", "mrr"),
                "seen_map": _metric(results, "seen", "map"),
                "seen_mrr": _metric(results, "seen", "mrr"),
                "best_epoch": history["best_epoch"],
                "best_val_loss": best_val_loss,
                "epochs_trained": len(train_loss_history),
                "stopped_early": history["stopped_early"],
                "seen_unseen_gap": seen_unseen_gap,
                "run_duration_sec": run_duration_sec,
            }
            for key, value in summary_values.items():
                wandb.summary[key] = value
            run_output = {
                "config": run_config,
                "input_dim": input_dim,
                "epochs_trained": len(train_loss_history),
                "best_epoch": history["best_epoch"],
                "best_val_loss": best_val_loss,
                "stopped_early": history["stopped_early"],
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
                "val_acc_history": val_acc_history,
                "results": results,
                "run_duration_sec": run_duration_sec,
                "wandb_run_id": getattr(run, "id", None),
                "wandb_run_url": getattr(run, "url", None),
            }
            _write_json(output_path, run_output)
            wandb.finish()
            run = None
            logger.info(
                "Run %d/%d: %s | best_epoch=%d | seen=%.3f unseen=%.3f all=%.3f",
                run_index,
                len(grid),
                run_name,
                history["best_epoch"],
                seen_ndcg1,
                unseen_ndcg1,
                all_ndcg1,
            )
        except Exception as error:
            logger.exception("Run %d/%d failed: %s", run_index, len(grid), run_name)
            if run is not None:
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    logger.exception("wandb.finish failed for %s", run_name)
            failures.append({"run_name": run_name, "error": str(error)})

    sorted_rows = _write_summary(
        output_dir,
        args.sweep_name,
        args.sweep_notes,
        len(grid),
        failures,
        run_names,
    )
    print("Run                              best_ep  seen_ndcg1  unseen_ndcg1  all_ndcg1  gap")
    for row in sorted_rows:
        print(
            f"{row['run_name']:<72} {row['best_epoch']:>7}"
            f"  {row['seen_ndcg1']:.3f}       {row['unseen_ndcg1']:.3f}"
            f"         {row['all_ndcg1']:.3f}    {row['seen_unseen_gap']:.3f}"
        )
    if failures:
        print("Failures:")
        for failure in failures:
            print(failure)


if __name__ == "__main__":
    main()
