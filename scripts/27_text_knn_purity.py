"""Measure text kNN closure-code purity for seen and unseen test incidents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logging_config import setup_logging
from src.sweep_runner import _prepare_dataframe, _set_seed, _split_positions


# Reference values from results/diagnostics/graph_structure.json.
BASE_RATE_REFERENCE = 0.314
CI_PATH_PURITY_SEEN = 0.342
UNSEEN_UNION_PURITY = 0.174


def compute_text_knn_purity(
    text_embeddings: np.ndarray,
    query_indices: np.ndarray,
    train_indices: np.ndarray,
    codes: np.ndarray,
    ks: list[int],
    chunk_size: int,
) -> dict[int, dict[str, float]]:
    """Measure closure-code agreement between queries and nearest train incidents."""
    if not ks or min(ks) < 1 or max(ks) > len(train_indices):
        raise ValueError("ks must be positive and no larger than training pool size")
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    normalized = np.divide(
        text_embeddings,
        norms,
        out=np.zeros_like(text_embeddings, dtype=np.float32),
        where=norms != 0,
    )
    candidates = normalized[train_indices]
    max_k = max(ks)
    purity_values = {k: [] for k in ks}
    top1_matches: list[bool] = []
    any_matches = {k: [] for k in ks}
    for start in range(0, len(query_indices), chunk_size):
        queries = query_indices[start : start + chunk_size]
        similarities = normalized[queries] @ candidates.T
        candidate_positions = np.argpartition(
            similarities, -max_k, axis=1
        )[:, -max_k:]
        selected_scores = np.take_along_axis(similarities, candidate_positions, axis=1)
        ordering = np.argsort(selected_scores, axis=1)[:, ::-1]
        nearest_positions = np.take_along_axis(candidate_positions, ordering, axis=1)
        neighbour_codes = codes[train_indices[nearest_positions]]
        matches = neighbour_codes == codes[queries, None]
        top1_matches.extend(matches[:, 0].tolist())
        for k in ks:
            per_query_matches = matches[:, :k].sum(axis=1)
            purity_values[k].extend((per_query_matches / k).tolist())
            any_matches[k].extend((per_query_matches > 0).tolist())
    return {
        k: {
            "mean_purity": float(np.mean(purity_values[k])),
            "median_purity": float(np.median(purity_values[k])),
            "std_purity": float(np.std(purity_values[k])),
            "top1_accuracy": float(np.mean(top1_matches)),
            "frac_queries_with_any_match": float(np.mean(any_matches[k])),
        }
        for k in ks
    }


def compute_random_baseline(
    query_indices: np.ndarray,
    train_indices: np.ndarray,
    codes: np.ndarray,
    ks: list[int],
    generator: np.random.Generator,
) -> dict[int, dict[str, float]]:
    """Measure purity of uniformly sampled training neighbours."""
    max_k = max(ks)
    matches_by_k = {k: [] for k in ks}
    top1_matches: list[bool] = []
    any_matches = {k: [] for k in ks}
    for start in range(0, len(query_indices), 512):
        queries = query_indices[start : start + 512]
        random_scores = generator.random((len(queries), len(train_indices)))
        positions = np.argpartition(random_scores, max_k - 1, axis=1)[:, :max_k]
        neighbour_codes = codes[train_indices[positions]]
        matches = neighbour_codes == codes[queries, None]
        top1_matches.extend(matches[:, 0].tolist())
        for k in ks:
            count = matches[:, :k].sum(axis=1)
            matches_by_k[k].extend((count / k).tolist())
            any_matches[k].extend((count > 0).tolist())
    return {
        k: {
            "mean_purity": float(np.mean(matches_by_k[k])),
            "median_purity": float(np.median(matches_by_k[k])),
            "std_purity": float(np.std(matches_by_k[k])),
            "top1_accuracy": float(np.mean(top1_matches)),
            "frac_queries_with_any_match": float(np.mean(any_matches[k])),
        }
        for k in ks
    }


def _sample_queries(
    indices: np.ndarray,
    max_queries: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample up to max_queries test indices without replacement."""
    if len(indices) <= max_queries:
        return np.sort(indices)
    return np.sort(generator.choice(indices, size=max_queries, replace=False))


def _plot_results(results: dict, ks: list[int], output_path: Path) -> None:
    """Save mean kNN purity and reference lines."""
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(ks, [results["test_seen"][k]["mean_purity"] for k in ks], marker="o", label="test_seen")
    axis.plot(ks, [results["test_unseen"][k]["mean_purity"] for k in ks], marker="o", label="test_unseen")
    axis.plot(ks, [results["random_baseline"][k]["mean_purity"] for k in ks], marker="o", label="random baseline")
    for value, label in (
        (BASE_RATE_REFERENCE, "base rate"),
        (CI_PATH_PURITY_SEEN, "CI-path seen"),
        (UNSEEN_UNION_PURITY, "unseen union"),
    ):
        axis.axhline(value, linestyle="--", label=f"{label} ({value:.3f})")
    axis.set_ylim(0, 1)
    axis.set_xlabel("k")
    axis.set_ylabel("Mean closure-code purity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run text kNN purity gate and write diagnostic JSON and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_queries < 1 or args.chunk_size < 1 or any(k < 1 for k in args.ks):
        raise ValueError("max-queries, chunk-size, and ks must be positive")
    args.ks = sorted(set(args.ks))
    logger = setup_logging("text_knn_purity")
    _set_seed(args.seed)
    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    logger.info("Text embedding shape: %s", text_embeddings.shape)
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test = test_indices[seen_mask.to_numpy()]
    unseen_test = test_indices[~seen_mask.to_numpy()]
    generator = np.random.default_rng(args.seed)
    queries = {
        "test_seen": _sample_queries(seen_test, args.max_queries, generator),
        "test_unseen": _sample_queries(unseen_test, args.max_queries, generator),
    }
    codes, _ = pd.factorize(dataframe["Closure Code"], sort=True)
    training_codes = codes[train_indices]
    base_rate = float(np.bincount(training_codes).max() / len(training_codes))
    logger.info("Training closure-code base rate: %.3f", base_rate)
    if args.dry_run:
        print(json.dumps({
            "config": vars(args),
            "embedding_shape": list(text_embeddings.shape),
            "candidate_pool_size": len(train_indices),
            "query_sample_sizes": {group: len(values) for group, values in queries.items()},
        }, indent=2))
        return
    results = {
        group: compute_text_knn_purity(
            text_embeddings, values, train_indices, codes, args.ks, args.chunk_size
        )
        for group, values in queries.items()
    }
    results["random_baseline"] = compute_random_baseline(
        np.concatenate(list(queries.values())), train_indices, codes, args.ks,
        np.random.default_rng(args.seed),
    )
    logger.info("k | seen mean | unseen mean | random | seen top1 | unseen top1")
    for k in args.ks:
        logger.info(
            "%d | %.3f | %.3f | %.3f | %.3f | %.3f",
            k,
            results["test_seen"][k]["mean_purity"],
            results["test_unseen"][k]["mean_purity"],
            results["random_baseline"][k]["mean_purity"],
            results["test_seen"][k]["top1_accuracy"],
            results["test_unseen"][k]["top1_accuracy"],
        )
        logger.info(
            "unseen k=%d mean %.3f vs base rate %.3f, CI-path seen %.3f, unseen union %.3f",
            k, results["test_unseen"][k]["mean_purity"], BASE_RATE_REFERENCE,
            CI_PATH_PURITY_SEEN, UNSEEN_UNION_PURITY,
        )
    output = {
        "config": vars(args),
        "embedding_shape": list(text_embeddings.shape),
        "base_rate": base_rate,
        "reference_points": {
            "base_rate": BASE_RATE_REFERENCE,
            "ci_path_purity_seen": CI_PATH_PURITY_SEEN,
            "unseen_union_purity": UNSEEN_UNION_PURITY,
        },
        "query_sample_sizes": {group: len(values) for group, values in queries.items()},
        "candidate_pool_size": len(train_indices),
        "results": results,
    }
    output_path = Path("results/diagnostics/text_knn_purity.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)
    if not args.skip_plots:
        try:
            figure_path = Path("results/diagnostics/figures/text_knn_purity.png")
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            _plot_results(results, args.ks, figure_path)
        except Exception:
            logger.exception("Plot generation failed after JSON was written")


if __name__ == "__main__":
    main()
