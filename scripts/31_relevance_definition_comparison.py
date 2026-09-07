"""Compare closure-first relevance definitions with a fixed text ranker."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from src.data_loader_sn import load_sn_incidents
from src.evaluate_retrieval import ndcg_at_k
from src.impute_subcategory import impute_subcategory
from src.logging_config import setup_logging
from src.relevance import compute_relevance_matrix

OBSERVED_UNSEEN_NDCG1 = 0.333
DEFINITIONS = ("A", "D", "E")
GROUPS = ("seen", "unseen", "all")
REQUIRED_COLUMNS = {
    "CI Name (aff)",
    "CI Subtype (aff)",
    "Closure Code",
    "Category",
}
RelevanceFunction = Callable[[pd.DataFrame, np.ndarray, np.ndarray], sparse.csr_matrix]


def _set_seed(seed: int) -> None:
    """Set reproducible Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def _prepare_dataframe(subtype: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load SN incidents and return closure-valid rows plus source positions."""
    dataframe = load_sn_incidents()
    original_indices = np.arange(len(dataframe), dtype=np.int64)
    if subtype == "imputed":
        dataframe = impute_subcategory(dataframe, int(len(dataframe) * 0.7))
    elif subtype == "none":
        dataframe["CI Subtype (aff)"] = np.nan
    valid_rows = dataframe["Closure Code"].notna()
    graph_row_indices = original_indices[valid_rows.to_numpy()]
    return dataframe.loc[valid_rows].reset_index(drop=True), graph_row_indices


def _split_positions(
    n_rows: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-overlapping positional train, validation, and test indices."""
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    return (
        np.arange(train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n_rows),
    )


def _valid_values(
    dataframe: pd.DataFrame,
    indices: np.ndarray,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return string values and original-dataframe validity for selected rows."""
    values = dataframe.iloc[indices][column]
    valid = ~values.isna().to_numpy()
    strings = values.astype("string").fillna("<missing>").to_numpy()
    return strings, valid


def _comparison(
    test_values: np.ndarray,
    train_values: np.ndarray,
    test_valid: np.ndarray,
    train_valid: np.ndarray,
) -> np.ndarray:
    """Compare values while excluding null values on either side."""
    return (
        test_valid[:, None]
        & train_valid[None, :]
        & (test_values[:, None] == train_values[None, :])
    )


def _definition_d(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> sparse.csr_matrix:
    """Build D: closure plus CI, subtype, or category graded relevance."""
    train_ci, train_ci_valid = _valid_values(dataframe, train_indices, "CI Name (aff)")
    test_ci, test_ci_valid = _valid_values(dataframe, test_indices, "CI Name (aff)")
    train_subtype, train_subtype_valid = _valid_values(
        dataframe, train_indices, "CI Subtype (aff)"
    )
    test_subtype, test_subtype_valid = _valid_values(
        dataframe, test_indices, "CI Subtype (aff)"
    )
    train_category, train_category_valid = _valid_values(
        dataframe, train_indices, "Category"
    )
    test_category, test_category_valid = _valid_values(
        dataframe, test_indices, "Category"
    )
    train_closure, train_closure_valid = _valid_values(
        dataframe, train_indices, "Closure Code"
    )
    test_closure, test_closure_valid = _valid_values(
        dataframe, test_indices, "Closure Code"
    )
    same_closure = _comparison(
        test_closure, train_closure, test_closure_valid, train_closure_valid
    )
    same_ci = _comparison(test_ci, train_ci, test_ci_valid, train_ci_valid)
    same_subtype = _comparison(
        test_subtype, train_subtype, test_subtype_valid, train_subtype_valid
    )
    same_category = _comparison(
        test_category, train_category, test_category_valid, train_category_valid
    )
    relevance = np.where(
        same_closure & same_ci,
        3,
        np.where(same_closure & same_subtype, 2, np.where(same_closure & same_category, 1, 0)),
    ).astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test_indices), len(train_indices)))


def _definition_e(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> sparse.csr_matrix:
    """Build E: closure plus CI, then subtype or category, then closure."""
    train_ci, train_ci_valid = _valid_values(dataframe, train_indices, "CI Name (aff)")
    test_ci, test_ci_valid = _valid_values(dataframe, test_indices, "CI Name (aff)")
    train_subtype, train_subtype_valid = _valid_values(
        dataframe, train_indices, "CI Subtype (aff)"
    )
    test_subtype, test_subtype_valid = _valid_values(
        dataframe, test_indices, "CI Subtype (aff)"
    )
    train_category, train_category_valid = _valid_values(
        dataframe, train_indices, "Category"
    )
    test_category, test_category_valid = _valid_values(
        dataframe, test_indices, "Category"
    )
    train_closure, train_closure_valid = _valid_values(
        dataframe, train_indices, "Closure Code"
    )
    test_closure, test_closure_valid = _valid_values(
        dataframe, test_indices, "Closure Code"
    )
    same_closure = _comparison(
        test_closure, train_closure, test_closure_valid, train_closure_valid
    )
    same_ci = _comparison(test_ci, train_ci, test_ci_valid, train_ci_valid)
    same_subtype = _comparison(
        test_subtype, train_subtype, test_subtype_valid, train_subtype_valid
    )
    same_category = _comparison(
        test_category, train_category, test_category_valid, train_category_valid
    )
    same_secondary = same_subtype | same_category
    relevance = np.where(
        same_closure & same_ci,
        3,
        np.where(same_closure & same_secondary, 2, np.where(same_closure, 1, 0)),
    ).astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test_indices), len(train_indices)))


def _definition_a(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> sparse.csr_matrix:
    """Build definition A explicitly despite production default D."""
    return compute_relevance_matrix(
        dataframe,
        train_indices,
        test_indices,
        definition="A",
    )


def _definitions() -> dict[str, RelevanceFunction]:
    """Return explicit A and local closure-first D/E definitions."""
    return {"A": _definition_a, "D": _definition_d, "E": _definition_e}


def _load_sweep_configs(sweep_dirs: Iterable[str]) -> dict[str, dict[str, object]]:
    """Read per-run configs from requested result sweep directories."""
    sweeps: dict[str, dict[str, object]] = {}
    for sweep_name in sweep_dirs:
        run_paths = sorted(Path("results/sweeps", sweep_name, "runs").glob("*.json"))
        runs = []
        for path in run_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "run_name": path.stem,
                    "config": payload.get("config", {}),
                }
            )
        sweeps[sweep_name] = {"num_runs": len(runs), "runs": runs}
    return sweeps


def _text_similarity_chunks(
    text_embeddings: np.ndarray,
    query_indices: np.ndarray,
    train_indices: np.ndarray,
    chunk_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield query index chunks and fixed text-ranker similarities."""
    query_embeddings = text_embeddings[query_indices]
    train_embeddings = text_embeddings[train_indices]
    query_embeddings = query_embeddings / np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    train_embeddings = train_embeddings / np.maximum(
        np.linalg.norm(train_embeddings, axis=1, keepdims=True), 1e-12
    )
    for start in range(0, len(query_indices), chunk_size):
        end = min(start + chunk_size, len(query_indices))
        yield (
            query_indices[start:end],
            query_embeddings[start:end] @ train_embeddings.T,
        )


def _row_metrics(relevance: np.ndarray, similarities: np.ndarray) -> dict[str, float]:
    """Compute fixed-ranker metrics for one query row."""
    order = np.argsort(-similarities, kind="stable")
    ranked = relevance[order]
    relevant = ranked > 0
    relevant_count = int(relevant.sum())
    if relevant_count:
        cumulative = np.cumsum(relevant)
        ranks = np.arange(1, len(relevant) + 1)
        average_precision = float((cumulative[relevant] / ranks[relevant]).sum() / relevant_count)
        reciprocal_rank = 1.0 / (int(np.flatnonzero(relevant)[0]) + 1)
    else:
        average_precision = 0.0
        reciprocal_rank = 0.0
    return {
        "nDCG@1": ndcg_at_k(ranked, 1),
        "nDCG@5": ndcg_at_k(ranked, 5),
        "MAP": average_precision,
        "MRR": reciprocal_rank,
    }


def _summarize_metrics(rows: list[dict[str, float]]) -> dict[str, object]:
    """Summarize per-query fixed-ranker metrics, including nDCG@1 spread."""
    if not rows:
        return {"num_queries": 0, **{name: None for name in ("nDCG@1", "nDCG@5", "MAP", "MRR")}, "ndcg@1_std": None}
    return {
        "num_queries": len(rows),
        **{
            name: float(np.mean([row[name] for row in rows]))
            for name in ("nDCG@1", "nDCG@5", "MAP", "MRR")
        },
        "ndcg@1_std": float(np.std([row["nDCG@1"] for row in rows])),
    }


def _process_group(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    text_embeddings: np.ndarray,
    definitions: dict[str, RelevanceFunction],
    chunk_size: int,
) -> tuple[dict[str, dict[str, object]], dict[str, np.ndarray]]:
    """Compute coverage and fixed-ranker metrics for one query group in chunks."""
    coverage: dict[str, dict[str, object]] = {}
    metric_rows: dict[str, list[dict[str, float]]] = {name: [] for name in DEFINITIONS}
    gradeable_rows: dict[str, list[dict[str, float]]] = {name: [] for name in DEFINITIONS}
    ndcg1_by_query: dict[str, list[float]] = {name: [] for name in DEFINITIONS}
    gradeable_by_query: dict[str, list[bool]] = {name: [] for name in DEFINITIONS}
    for definition in DEFINITIONS:
        coverage[definition] = {
            "num_queries": 0,
            "all_zero_count": 0,
            "positive_counts": [],
            "grade_counts": {"3": [], "2": [], "1": []},
            "joint_null_count": 0,
        }

    for chunk_indices, similarities in _text_similarity_chunks(
        text_embeddings, query_indices, train_indices, chunk_size
    ):
        subtype_null = dataframe.iloc[chunk_indices]["CI Subtype (aff)"].isna().to_numpy()
        category_null = dataframe.iloc[chunk_indices]["Category"].isna().to_numpy()
        for definition, relevance_fn in definitions.items():
            relevance = relevance_fn(dataframe, train_indices, chunk_indices).toarray()
            stats = coverage[definition]
            stats["num_queries"] += len(chunk_indices)
            stats["all_zero_count"] += int(np.sum(np.all(relevance == 0, axis=1)))
            stats["joint_null_count"] += int(np.sum(subtype_null & category_null))
            stats["positive_counts"].extend(np.sum(relevance > 0, axis=1).tolist())
            for grade in (3, 2, 1):
                stats["grade_counts"][str(grade)].extend(
                    np.sum(relevance == grade, axis=1).tolist()
                )
            for row_relevance, row_similarity in zip(relevance, similarities):
                row_metrics = _row_metrics(row_relevance, row_similarity)
                metric_rows[definition].append(row_metrics)
                ndcg1_by_query[definition].append(row_metrics["nDCG@1"])
            gradeable = np.any(relevance > 0, axis=1)
            gradeable_by_query[definition].extend(gradeable.tolist())
            gradeable_rows[definition].extend(
                row
                for row, is_gradeable in zip(
                    metric_rows[definition][-len(relevance) :], gradeable
                )
                if is_gradeable
            )

    result: dict[str, dict[str, object]] = {}
    for definition in DEFINITIONS:
        stats = coverage[definition]
        positive_counts = np.asarray(stats.pop("positive_counts"), dtype=float)
        grade_counts = stats.pop("grade_counts")
        num_queries = int(stats["num_queries"])
        result[definition] = {
            "num_queries": num_queries,
            "all_zero_fraction": stats["all_zero_count"] / num_queries if num_queries else 0.0,
            "implied_ndcg1_ceiling": 1.0 - stats["all_zero_count"] / num_queries if num_queries else 0.0,
            "mean_positive_candidates": float(np.mean(positive_counts)) if num_queries else 0.0,
            "median_positive_candidates": float(np.median(positive_counts)) if num_queries else 0.0,
            "mean_count_grade_3": float(np.mean(grade_counts["3"])) if num_queries else 0.0,
            "mean_count_grade_2": float(np.mean(grade_counts["2"])) if num_queries else 0.0,
            "mean_count_grade_1": float(np.mean(grade_counts["1"])) if num_queries else 0.0,
            "joint_null_rate": stats["joint_null_count"] / num_queries if num_queries else 0.0,
            "text_ranker": {
                "all": _summarize_metrics(metric_rows[definition]),
                "gradeable": _summarize_metrics(gradeable_rows[definition]),
            },
        }
    return result, {
        definition: np.asarray(ndcg1_by_query[definition], dtype=float)
        for definition in DEFINITIONS
    } | {
        f"{definition}_gradeable": np.asarray(gradeable_by_query[definition], dtype=bool)
        for definition in DEFINITIONS
    }


def _log_coverage(logger: logging.Logger, coverage: dict[str, dict[str, dict[str, object]]]) -> None:
    """Log Part 1 coverage table."""
    logger.info("=== Part 1: coverage and gradeability ===")
    logger.info("definition group all_zero ceiling pos_mean pos_median grade3_mean grade2_mean grade1_mean joint_null")
    for definition in DEFINITIONS:
        for group in ("seen", "unseen"):
            stats = coverage[definition][group]
            logger.info(
                "%s %s %.3f %.3f %.1f %.1f %.1f %.1f %.1f %.3f%s",
                definition,
                group,
                stats["all_zero_fraction"],
                stats["implied_ndcg1_ceiling"],
                stats["mean_positive_candidates"],
                stats["median_positive_candidates"],
                stats["mean_count_grade_3"],
                stats["mean_count_grade_2"],
                stats["mean_count_grade_1"],
                stats["joint_null_rate"],
                f" ceiling_vs_observed_unseen={stats['implied_ndcg1_ceiling']:.3f} vs 0.333"
                if group == "unseen"
                else "",
            )


def _log_metrics(logger: logging.Logger, metrics: dict[str, dict[str, dict[str, object]]]) -> None:
    """Log Part 2 fixed-ranker metric table."""
    logger.info("=== Part 2: fixed text-ranker discrimination ===")
    logger.info("definition group subset nDCG@1 nDCG@5 MAP MRR nDCG@1_std")
    for definition in DEFINITIONS:
        for group in ("seen", "unseen"):
            for subset in ("all", "gradeable"):
                stats = metrics[definition][group]["text_ranker"][subset]
                logger.info(
                    "%s %s %s %s %s %s %s %s",
                    definition,
                    group,
                    subset,
                    *(f"{stats[name]:.3f}" if stats[name] is not None else "N/A" for name in ("nDCG@1", "nDCG@5", "MAP", "MRR", "ndcg@1_std")),
                )
        gradeable_seen = metrics[definition]["seen"]["text_ranker"]["gradeable"]
        gradeable_unseen = metrics[definition]["unseen"]["text_ranker"]["gradeable"]
        gap = {
            name: gradeable_seen[name] - gradeable_unseen[name]
            for name in ("nDCG@1", "nDCG@5", "MAP", "MRR")
        }
        logger.info("%s gradeable seen-minus-unseen gap %s", definition, gap)
        if gradeable_seen["nDCG@1"] is not None and gradeable_seen["nDCG@1"] > 0.8:
            logger.warning("%s flagged: seen gradeable text-ranker nDCG@1 > 0.8", definition)
        if gradeable_unseen["nDCG@1"] is not None and gradeable_unseen["nDCG@1"] > 0.8:
            logger.warning("%s flagged: unseen gradeable text-ranker nDCG@1 > 0.8", definition)


def _log_correlations(logger: logging.Logger, correlations: dict[str, object]) -> None:
    """Log Part 3 rank-correlation table."""
    logger.info("=== Part 3: unseen gradeable per-query nDCG@1 Spearman correlation ===")
    logger.info("comparison correlation")
    for comparison, value in correlations.items():
        logger.info("%s %s", comparison, f"{value:.3f}" if value is not None else "N/A")


def _plot_results(
    coverage: dict[str, dict[str, dict[str, object]]],
    metrics: dict[str, dict[str, dict[str, object]]],
    output_path: Path,
) -> None:
    """Save ceiling and fixed-ranker gradeable nDCG@1 panels."""
    labels = list(DEFINITIONS)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.35
    for offset, group in enumerate(("seen", "unseen")):
        values = [coverage[definition][group]["implied_ndcg1_ceiling"] for definition in labels]
        axes[0].bar(x + (offset - 0.5) * width, values, width, label=group)
    axes[0].axhline(OBSERVED_UNSEEN_NDCG1, color="black", linestyle="--", label="observed unseen 0.333")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Availability ceiling")
    axes[0].set_ylabel("Implied nDCG@1 ceiling")
    axes[0].legend()
    values = [
        metrics[definition]["seen"]["text_ranker"]["gradeable"]["nDCG@1"]
        for definition in labels
    ]
    axes[1].bar(x, values, width, label="gradeable text ranker")
    axes[1].axhline(0.8, color="black", linestyle="--", label="permissiveness flag 0.8")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Fixed text-ranker nDCG@1")
    axes[1].set_ylabel("nDCG@1")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--sweep-dirs",
        nargs="+",
        default=["features_seeds_v2", "concat_embed_dim_seeds_v2"],
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Run comparison, write JSON before optional plotting, and log tables."""
    args = _build_parser().parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    logger = setup_logging("relevance_definition_comparison")
    _set_seed(args.seed)
    sweep_configs = _load_sweep_configs(args.sweep_dirs)
    dataframe, graph_row_indices = _prepare_dataframe(args.subtype)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    train_indices, _, test_indices = _split_positions(len(dataframe))
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    groups = {
        "seen": test_indices[seen_mask.to_numpy()],
        "unseen": test_indices[~seen_mask.to_numpy()],
        "all": test_indices,
    }
    if args.dry_run:
        print("configuration:")
        print(json.dumps(vars(args), indent=2))
        print("sweep_runs:")
        print(json.dumps({name: value["num_runs"] for name, value in sweep_configs.items()}, indent=2))
        print("group_sizes:")
        print(json.dumps({name: len(indices) for name, indices in groups.items()}, indent=2))
        print(f"train_size: {len(train_indices)}")
        return

    text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    definitions = _definitions()
    coverage: dict[str, dict[str, dict[str, object]]] = {definition: {} for definition in DEFINITIONS}
    query_arrays: dict[str, dict[str, np.ndarray]] = {definition: {} for definition in DEFINITIONS}
    for group in GROUPS:
        group_coverage, group_arrays = _process_group(
            dataframe,
            train_indices,
            groups[group],
            text_embeddings,
            definitions,
            args.chunk_size,
        )
        for definition in DEFINITIONS:
            coverage[definition][group] = group_coverage[definition]
            query_arrays[definition][group] = group_arrays[definition]
            query_arrays[definition][f"{group}_gradeable"] = group_arrays[f"{definition}_gradeable"]

    _log_coverage(logger, coverage)
    _log_metrics(logger, coverage)
    correlations: dict[str, object] = {}
    common_gradeable = np.logical_and.reduce(
        [query_arrays[definition]["unseen_gradeable"] for definition in DEFINITIONS]
    )
    for left, right in (("A", "D"), ("A", "E"), ("D", "E")):
        if np.sum(common_gradeable) < 2:
            correlations[f"{left}/{right}"] = None
        else:
            correlations[f"{left}/{right}"] = float(
                spearmanr(
                    query_arrays[left]["unseen"][common_gradeable],
                    query_arrays[right]["unseen"][common_gradeable],
                ).statistic
            )
    _log_correlations(logger, correlations)

    output = {
        "config": vars(args),
        "sweep_configs": sweep_configs,
        "group_sizes": {name: len(indices) for name, indices in groups.items()},
        "measurement_a_coverage": coverage,
        "measurement_b_fixed_text_ranker": {
            definition: {
                group: coverage[definition][group]["text_ranker"]
                for group in ("seen", "unseen")
            }
            for definition in DEFINITIONS
        },
        "measurement_b_gradeable_seen_minus_unseen_gap": {
            definition: {
                name: coverage[definition]["seen"]["text_ranker"]["gradeable"][name]
                - coverage[definition]["unseen"]["text_ranker"]["gradeable"][name]
                for name in ("nDCG@1", "nDCG@5", "MAP", "MRR")
            }
            for definition in DEFINITIONS
        },
        "measurement_c_spearman_unseen_gradeable_common_subset": {
            "num_queries": int(np.sum(common_gradeable)),
            "correlations": correlations,
        },
    }
    output_path = Path("results/diagnostics/relevance_definition_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)
    if not args.skip_plots:
        figure_path = Path("results/diagnostics/figures/relevance_definitions.png")
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_results(coverage, coverage, figure_path)
        logger.info("Plot saved: %s", figure_path)


if __name__ == "__main__":
    main()
