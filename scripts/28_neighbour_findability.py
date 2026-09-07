"""Measure existence and text-similarity rank of high-relevance training neighbours."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.logging_config import setup_logging
from src.relevance import compute_relevance_matrix
from src.sweep_runner import _prepare_dataframe, _set_seed, _split_positions


def _sample_queries(
    indices: np.ndarray,
    max_queries: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample up to max_queries indices without replacement and sort them."""
    if len(indices) <= max_queries:
        return np.sort(indices)
    return np.sort(generator.choice(indices, size=max_queries, replace=False))


def _random_baseline_hit_rate(pool_size: int, good_count: int, cutoff: int) -> float:
    """Compute expected hit rate at cutoff under random ranking."""
    if good_count == 0:
        return 0.0
    if cutoff >= pool_size:
        return 1.0 if good_count > 0 else 0.0
    prob_miss = 1.0
    for i in range(cutoff):
        prob_miss *= (pool_size - good_count - i) / (pool_size - i)
    return 1.0 - prob_miss


def compute_findability(
    text_embeddings: np.ndarray,
    query_indices: np.ndarray,
    train_indices: np.ndarray,
    relevance_fn: callable,
    threshold: int,
    rank_cutoffs: list[int],
    chunk_size: int,
) -> dict:
    """Measure existence and text-similarity rank of high-relevance training neighbours.

    For each query, counts training incidents at or above the relevance
    threshold, then ranks all training incidents by cosine text similarity and
    records the rank of the first qualifying one.

    Args:
        text_embeddings: Array of shape (num_incidents, dim), dataframe-aligned.
        query_indices: Query incident row indices.
        train_indices: Candidate training incident row indices.
        relevance_fn: Callable returning the graded relevance row for a query
            against all train candidates.
        threshold: Minimum relevance counting as a good neighbour.
        rank_cutoffs: Rank positions at which to report hit rates.
        chunk_size: Query rows per chunk.

    Returns:
        Dict with num_good_neighbours stats, first_hit_rank stats, per-cutoff
        hit rates, and the fraction of queries with zero good neighbours.
    """
    query_embeddings = text_embeddings[query_indices]
    train_embeddings = text_embeddings[train_indices]
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / np.maximum(query_norms, 1e-12)
    train_embeddings = train_embeddings / np.maximum(train_norms, 1e-12)
    num_good_neighbours = []
    first_hit_ranks = []
    hits_at_cutoff = {cutoff: 0 for cutoff in rank_cutoffs}
    relevance_matrix = relevance_fn(query_indices, train_indices)
    good_mask = relevance_matrix >= threshold
    similarities = query_embeddings @ train_embeddings.T
    for i in range(len(query_indices)):
        good_indices = np.flatnonzero(good_mask[i])
        num_good = len(good_indices)
        num_good_neighbours.append(num_good)
        if num_good == 0:
            continue
        ranked_indices = np.argsort(-similarities[i])
        first_hit_position = None
        for rank, train_position in enumerate(ranked_indices):
            if good_mask[i, train_position]:
                first_hit_position = rank + 1
                break
        if first_hit_position is not None:
            first_hit_ranks.append(first_hit_position)
            for cutoff in rank_cutoffs:
                if first_hit_position <= cutoff:
                    hits_at_cutoff[cutoff] += 1
    num_good_neighbours = np.array(num_good_neighbours, dtype=np.int64)
    first_hit_ranks = np.array(first_hit_ranks, dtype=np.int64)
    zero_fraction = float(np.mean(num_good_neighbours == 0))
    num_good_stats = {
        "mean": float(np.mean(num_good_neighbours)),
        "median": float(np.median(num_good_neighbours)),
        "q25": float(np.percentile(num_good_neighbours, 25)),
        "q75": float(np.percentile(num_good_neighbours, 75)),
    }
    if len(first_hit_ranks) > 0:
        first_hit_stats = {
            "median": float(np.median(first_hit_ranks)),
            "q25": float(np.percentile(first_hit_ranks, 25)),
            "q75": float(np.percentile(first_hit_ranks, 75)),
        }
    else:
        first_hit_stats = {"median": None, "q25": None, "q75": None}
    hit_rates = {
        cutoff: float(hits_at_cutoff[cutoff]) / len(query_indices)
        for cutoff in rank_cutoffs
    }
    mean_good = float(np.mean(num_good_neighbours))
    random_baseline = {
        cutoff: _random_baseline_hit_rate(len(train_indices), int(mean_good), cutoff)
        for cutoff in rank_cutoffs
    }
    return {
        "zero_good_fraction": zero_fraction,
        "num_good_neighbours": num_good_stats,
        "first_hit_rank": first_hit_stats,
        "hit_rates": hit_rates,
        "random_baseline": random_baseline,
    }


def _plot_findability(results: dict, rank_cutoffs: list[int], output_path: Path) -> None:
    """Save hit rate versus rank cutoff plot."""
    figure, axis = plt.subplots(figsize=(10, 6))
    for group in ["test_seen", "test_unseen"]:
        for threshold_key in results[group]:
            threshold = int(threshold_key.split("_")[1])
            hit_rates = results[group][threshold_key]["hit_rates"]
            cutoffs = sorted(hit_rates.keys())
            rates = [hit_rates[c] for c in cutoffs]
            label = f"{group.replace('test_', '')} rel≥{threshold}"
            axis.plot(cutoffs, rates, marker="o", label=label)
    baseline_key = list(results["test_seen"].keys())[0]
    baseline = results["test_seen"][baseline_key]["random_baseline"]
    cutoffs = sorted(baseline.keys())
    rates = [baseline[c] for c in cutoffs]
    axis.plot(cutoffs, rates, linestyle="--", color="gray", label="random baseline")
    axis.set_xscale("log")
    axis.set_xlabel("Rank cutoff")
    axis.set_ylabel("Hit rate")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run neighbour findability analysis and save metrics and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--max-queries", type=int, default=1449)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--relevance-thresholds", nargs="+", type=int, default=[3, 2])
    parser.add_argument("--rank-cutoffs", nargs="+", type=int, default=[1, 5, 10, 50, 100, 500, 1000, 5000])
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("neighbour_findability")
    _set_seed(args.seed)
    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_test_mask = test_cis.notna() & test_cis.isin(train_cis)
    test_seen = test_indices[seen_test_mask.to_numpy()]
    test_unseen = test_indices[~seen_test_mask.to_numpy()]
    generator = np.random.default_rng(args.seed)
    test_unseen_sampled = _sample_queries(test_unseen, args.max_queries, generator)
    test_seen_sampled = _sample_queries(test_seen, len(test_unseen_sampled), generator)
    test_groups = {
        "test_seen": test_seen_sampled,
        "test_unseen": test_unseen_sampled,
    }
    if args.dry_run:
        print("configuration:")
        print(json.dumps(vars(args), indent=2))
        print("query_sample_sizes:")
        print(json.dumps({group: len(indices) for group, indices in test_groups.items()}, indent=2))
        print(f"train_size: {len(train_indices)}")
        print(f"text_embeddings_shape: {text_embeddings.shape}")
        return

    def relevance_fn(query_indices, candidate_indices):
        return compute_relevance_matrix(dataframe, candidate_indices, query_indices).toarray()

    results = {}
    for group, query_indices in test_groups.items():
        results[group] = {}
        for threshold in args.relevance_thresholds:
            logger.info(
                "Computing findability for %s with relevance threshold %d (%d queries)",
                group, threshold, len(query_indices),
            )
            findability = compute_findability(
                text_embeddings,
                query_indices,
                train_indices,
                relevance_fn,
                threshold,
                args.rank_cutoffs,
                args.chunk_size,
            )
            results[group][f"threshold_{threshold}"] = findability
            logger.info(
                "%s rel≥%d: zero_good=%.3f, mean_good=%.1f, median_good=%.1f, median_rank=%s",
                group, threshold,
                findability["zero_good_fraction"],
                findability["num_good_neighbours"]["mean"],
                findability["num_good_neighbours"]["median"],
                findability["first_hit_rank"]["median"],
            )
    output = {
        "config": vars(args),
        "query_sample_sizes": {group: len(indices) for group, indices in test_groups.items()},
        "train_size": len(train_indices),
        "results": results,
    }
    output_path = Path("results/diagnostics/neighbour_findability.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)
    if not args.skip_plots:
        figures_dir = Path("results/diagnostics/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_findability(results, args.rank_cutoffs, figures_dir / "neighbour_findability.png")
        except Exception:
            logger.exception("Plot generation failed after JSON was written")


if __name__ == "__main__":
    main()
