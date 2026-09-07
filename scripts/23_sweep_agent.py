"""Create and execute native wandb sweeps for SN incident retrieval."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from pprint import pformat

import wandb

from src.sweep_runner import (
    RESULTS_ROOT,
    WANDB_PROJECT,
    _git_commit,
    _write_json,
    build_run_name,
    run_one,
    write_sweep_summaries,
)


LOGGER = logging.getLogger("sweep_agent")


def build_sweep_config(args: argparse.Namespace) -> dict:
    """Build wandb sweep configuration from parsed arguments."""
    return {
        "name": args.sweep_name,
        "method": args.method,
        "metric": {"name": args.metric_name, "goal": args.metric_goal},
        "parameters": {
            "features": {"values": [str(v) for v in args.features]},
            "entity_embed_dim": {"values": [int(v) for v in args.entity_embed_dims]},
            "ci_dropout_rate": {"values": [float(v) for v in args.ci_dropout_rates]},
            "ci_dropout_mode": {"values": [str(v) for v in args.ci_dropout_modes]},
            "lr": {"values": [float(v) for v in args.lrs]},
            "weight_decay": {"values": [float(v) for v in args.weight_decays]},
            "hidden_dim": {"values": [int(v) for v in args.hidden_dims]},
            "num_layers": {"values": [int(v) for v in args.num_layers_list]},
            "dropout": {"values": [float(v) for v in args.dropouts]},
            "triplet_weight": {"values": [float(v) for v in args.triplet_weights]},
            "triplet_margin": {"values": [float(v) for v in args.triplet_margins]},
            "seed": {"values": [int(v) for v in args.seeds]},
            "knn_edges": {"values": [int(v) for v in args.knn_edges]},
            "knn_scope": {"values": [str(v) for v in args.knn_scopes]},
        },
    }


def make_train_fn(args: argparse.Namespace, git_commit: str | None):
    """Build zero-argument training callable for wandb.agent."""
    def train() -> None:
        """Initialize, execute, and finish one agent-managed wandb run."""
        try:
            wandb.init(
                group=args.sweep_name,
                job_type="sweep_run",
                notes=args.sweep_notes,
            )
            config = {
                key: wandb.config[key]
                for key in (
                    "features", "entity_embed_dim", "ci_dropout_rate",
                    "ci_dropout_mode", "lr", "weight_decay", "hidden_dim",
                    "num_layers", "dropout", "triplet_weight", "triplet_margin",
                    "seed", "knn_edges", "knn_scope",
                )
            }
            run_name = build_run_name(config)
            wandb.run.name = run_name
            run_one(
                config,
                args.sweep_name,
                args.sweep_notes,
                args.epochs,
                args.patience,
                args.subtype,
                run_name=run_name,
            )
        except Exception:
            LOGGER.exception("Sweep run failed")
            raise
        finally:
            wandb.finish()

    return train


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sweep-agent command-line arguments to parser."""
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--sweep-notes", default="")
    parser.add_argument("--entity-embed-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--ci-dropout-rates", nargs="+", type=float, default=[0.5])
    parser.add_argument("--lrs", nargs="+", type=float, default=[5e-4])
    parser.add_argument("--features", nargs="+", choices=["categorical", "text", "concat"], default=["concat"])
    parser.add_argument("--ci-dropout-modes", nargs="+", choices=["embedding", "edge", "both"], default=["embedding"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--num-layers-list", nargs="+", type=int, default=[2])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3])
    parser.add_argument("--triplet-weights", nargs="+", type=float, default=[0.0])
    parser.add_argument("--triplet-margins", nargs="+", type=float, default=[0.2])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--knn-edges", nargs="+", type=int, default=[0])
    parser.add_argument("--knn-scopes", nargs="+", choices=["all", "unseen_only"], default=["all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--method", choices=["grid", "random", "bayes"], default="grid")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--metric-name", default="unseen_ndcg1")
    parser.add_argument("--metric-goal", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser.parse_args()


def _yaml_style(value: object, indent: int = 0) -> list[str]:
    """Render basic dictionaries and lists as YAML-style text."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_style(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_style(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
        return lines
    return [f"{prefix}{value}"]


def _sweep_url(sweep_id: str) -> str:
    """Build best-effort wandb sweep URL from configured entity."""
    entity = os.getenv("WANDB_ENTITY", "")
    prefix = f"{entity}/" if entity else ""
    return f"https://wandb.ai/{prefix}{WANDB_PROJECT}/sweeps/{sweep_id}"


def _manifest_path(output_dir: Path) -> Path:
    """Choose manifest.json or timestamped fallback without overwriting existing data."""
    path = output_dir / "manifest.json"
    if not path.exists():
        return path
    return output_dir / f"manifest_{datetime.now():%Y%m%d_%H%M%S}.json"


def _print_tables(rows: list[dict], config_rows: list[dict]) -> None:
    """Print per-run and per-config summary tables."""
    print("Run                              best_ep  seen_ndcg1  unseen_ndcg1  all_ndcg1  gap")
    for row in rows:
        print(
            f"{row['run_name']:<72} {row['best_epoch']:>7}"
            f"  {row['seen_ndcg1']:.3f}       {row['unseen_ndcg1']:.3f}"
            f"         {row['all_ndcg1']:.3f}    {row['seen_unseen_gap']:.3f}"
        )
    print("Config                           seeds  seen_ndcg1  unseen_ndcg1  all_ndcg1  best_epoch")
    for row in config_rows:
        print(
            f"{row['config_label']:<32} {row['n_seeds']:>5}"
            f"  {row['seen_ndcg1_mean']:.3f}       {row['unseen_ndcg1_mean']:.3f}"
            f"         {row['all_ndcg1_mean']:.3f}    {row['best_epoch_mean']:.3f}"
        )


def main() -> None:
    """Parse arguments, create or attach to sweep, and run agent."""
    args = parse_args()
    sweep_config = build_sweep_config(args)
    if args.method in {"random", "bayes"} and args.count is None:
        raise SystemExit("--count is required for random and bayes methods")
    if args.dry_run:
        print("sweep_config:")
        print("\n".join(_yaml_style(sweep_config, 2)))
        if args.method == "grid":
            parameter_values = list(sweep_config["parameters"].values())
            total = 1
            for parameter in parameter_values:
                total *= len(parameter["values"])
            print(f"total_grid_runs: {total}")
        return

    output_dir = Path(RESULTS_ROOT) / args.sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    git_commit = _git_commit()
    timestamp_start = datetime.now(timezone.utc).isoformat()
    manifest = {
        "sweep_name": args.sweep_name,
        "sweep_notes": args.sweep_notes,
        "timestamp_start": timestamp_start,
        "git_commit": git_commit,
        "axes": {
            name: parameter["values"]
            for name, parameter in sweep_config["parameters"].items()
        },
        "method": args.method,
        "count": args.count,
        "metric": sweep_config["metric"],
        "epochs": args.epochs,
        "patience": args.patience,
        "subtype": args.subtype,
    }
    sweep_id = args.sweep_id or wandb.sweep(sweep_config, project=WANDB_PROJECT)
    sweep_url = _sweep_url(sweep_id)
    manifest.update({"sweep_id": sweep_id, "sweep_url": sweep_url})
    _write_json(_manifest_path(output_dir), manifest)
    LOGGER.info("Sweep ID: %s", sweep_id)
    LOGGER.info("Sweep URL: %s", sweep_url)
    wandb.agent(
        sweep_id,
        function=make_train_fn(args, git_commit),
        project=WANDB_PROJECT,
        count=args.count,
    )
    rows, config_rows = write_sweep_summaries(
        args.sweep_name, args.sweep_notes, args.count or 0
    )
    _print_tables(rows, config_rows)


if __name__ == "__main__":
    main()
