"""Audit relevance availability and its impact on retrieval ceilings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from src.evaluate_retrieval import evaluate_retrieval_graded
from src.logging_config import setup_logging
from src.relevance import compute_relevance_matrix
from src.sweep_runner import _prepare_dataframe, _set_seed, _split_positions


OBSERVED_SEEN_NDCG1 = 0.691
OBSERVED_UNSEEN_NDCG1 = 0.333


def _relevance_definition_a(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> sparse.csr_matrix:
    """Current definition: 3=same CI+closure, 2=same CI, 1=same subtype+closure."""
    return compute_relevance_matrix(df, train_indices, test_indices)


def _relevance_definition_b(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> sparse.csr_matrix:
    """Closure-first: 3=same CI+closure, 2=same closure+subtype, 1=same closure."""
    train_values = df.iloc[train_indices]
    test_values = df.iloc[test_indices]
    train_ci = train_values["CI Name (aff)"].astype("string").fillna("<missing>").to_numpy()
    test_ci = test_values["CI Name (aff)"].astype("string").fillna("<missing>").to_numpy()
    train_subtype = train_values["CI Subtype (aff)"].astype("string").fillna("<missing>").to_numpy()
    test_subtype = test_values["CI Subtype (aff)"].astype("string").fillna("<missing>").to_numpy()
    train_closure = train_values["Closure Code"].astype("string").fillna("<missing>").to_numpy()
    test_closure = test_values["Closure Code"].astype("string").fillna("<missing>").to_numpy()
    same_ci = test_ci[:, None] == train_ci[None, :]
    same_subtype = test_subtype[:, None] == train_subtype[None, :]
    same_closure = test_closure[:, None] == train_closure[None, :]
    relevance = np.where(
        same_ci & same_closure,
        3,
        np.where(same_closure & same_subtype, 2, np.where(same_closure, 1, 0)),
    ).astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test_indices), len(train_indices)))


def _relevance_definition_c(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> sparse.csr_matrix:
    """Resolution-only: 1=same closure code, 0 otherwise."""
    train_values = df.iloc[train_indices]
    test_values = df.iloc[test_indices]
    train_closure = train_values["Closure Code"].astype("string").fillna("<missing>").to_numpy()
    test_closure = test_values["Closure Code"].astype("string").fillna("<missing>").to_numpy()
    same_closure = test_closure[:, None] == train_closure[None, :]
    relevance = same_closure.astype(np.int8, copy=False)
    return sparse.csr_matrix(relevance, shape=(len(test_indices), len(train_indices)))


def _chunked_relevance(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    relevance_fn: callable,
    chunk_size: int,
) -> sparse.csr_matrix:
    """Build relevance matrix in query chunks to limit peak memory."""
    chunks = []
    for start in range(0, len(test_indices), chunk_size):
        end = min(start + chunk_size, len(test_indices))
        chunk = relevance_fn(df, train_indices, test_indices[start:end])
        chunks.append(chunk)
    return sparse.vstack(chunks, format="csr")


def _measure_availability(
    relevance: sparse.csr_matrix,
) -> dict:
    """Measure relevance availability stats from a relevance matrix."""
    relevance_dense = relevance.toarray().astype(np.int8)
    num_queries = relevance_dense.shape[0]
    all_zero = np.all(relevance_dense == 0, axis=1)
    has_rel1 = np.any(relevance_dense >= 1, axis=1)
    has_rel2 = np.any(relevance_dense >= 2, axis=1)
    has_rel3 = np.any(relevance_dense == 3, axis=1)
    counts_rel1 = np.sum(relevance_dense >= 1, axis=1)
    counts_rel2 = np.sum(relevance_dense >= 2, axis=1)
    counts_rel3 = np.sum(relevance_dense == 3, axis=1)
    return {
        "num_queries": int(num_queries),
        "all_zero_fraction": float(np.mean(all_zero)),
        "has_rel1_fraction": float(np.mean(has_rel1)),
        "has_rel2_fraction": float(np.mean(has_rel2)),
        "has_rel3_fraction": float(np.mean(has_rel3)),
        "mean_count_rel1": float(np.mean(counts_rel1)),
        "median_count_rel1": float(np.median(counts_rel1)),
        "mean_count_rel2": float(np.mean(counts_rel2)),
        "median_count_rel2": float(np.median(counts_rel2)),
        "mean_count_rel3": float(np.mean(counts_rel3)),
        "median_count_rel3": float(np.median(counts_rel3)),
        "implied_ndcg1_ceiling": float(1.0 - np.mean(all_zero)),
    }


def _measure_subtype_null(
    df: pd.DataFrame,
    relevance: sparse.csr_matrix,
    test_indices: np.ndarray,
) -> dict:
    """Measure subtype null interaction for unseen queries."""
    subtypes = df.iloc[test_indices]["CI Subtype (aff)"]
    is_null = subtypes.isna().to_numpy()
    relevance_dense = relevance.toarray().astype(np.int8)
    all_zero = np.all(relevance_dense == 0, axis=1)
    has_rel1 = np.any(relevance_dense >= 1, axis=1)
    null_count = int(np.sum(is_null))
    nonnull_count = int(np.sum(~is_null))
    return {
        "num_null_subtype": null_count,
        "fraction_null_subtype": float(np.mean(is_null)) if len(is_null) else 0.0,
        "nonnull_has_rel1_fraction": float(np.mean(has_rel1[~is_null])) if nonnull_count else 0.0,
        "null_all_zero_fraction": float(np.mean(all_zero[is_null])) if null_count else None,
    }


def _text_similarity_ranker(
    text_embeddings: np.ndarray,
    query_indices: np.ndarray,
    train_indices: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Compute cosine similarity matrix between queries and train candidates."""
    query_emb = text_embeddings[query_indices]
    train_emb = text_embeddings[train_indices]
    query_norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
    train_norms = np.linalg.norm(train_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.maximum(query_norms, 1e-12)
    train_emb = train_emb / np.maximum(train_norms, 1e-12)
    similarities = np.zeros((len(query_indices), len(train_indices)), dtype=np.float32)
    for start in range(0, len(query_indices), chunk_size):
        end = min(start + chunk_size, len(query_indices))
        similarities[start:end] = query_emb[start:end] @ train_emb.T
    return similarities


def _plot_ceiling(
    ceiling_results: dict,
    output_path: Path,
) -> None:
    """Save grouped bar chart of implied nDCG@1 ceiling per group per definition."""
    definitions = ["A_current", "B_closure_first", "C_resolution_only"]
    groups = ["seen", "unseen"]
    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(definitions))
    width = 0.35
    seen_ceilings = [ceiling_results[d]["seen"]["implied_ndcg1_ceiling"] for d in definitions]
    unseen_ceilings = [ceiling_results[d]["unseen"]["implied_ndcg1_ceiling"] for d in definitions]
    axis.bar(x - width / 2, seen_ceilings, width, label="seen")
    axis.bar(x + width / 2, unseen_ceilings, width, label="unseen")
    axis.axhline(OBSERVED_SEEN_NDCG1, color="#1f77b4", linestyle="--", alpha=0.7, label=f"observed seen {OBSERVED_SEEN_NDCG1:.3f}")
    axis.axhline(OBSERVED_UNSEEN_NDCG1, color="#ff7f0e", linestyle="--", alpha=0.7, label=f"observed unseen {OBSERVED_UNSEEN_NDCG1:.3f}")
    axis.set_xticks(x, definitions, rotation=15)
    axis.set_ylabel("Implied nDCG@1 ceiling")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run relevance audit and save measurements and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("relevance_audit")
    _set_seed(args.seed)
    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_test_mask = test_cis.notna() & test_cis.isin(train_cis)
    test_seen = test_indices[seen_test_mask.to_numpy()]
    test_unseen = test_indices[~seen_test_mask.to_numpy()]
    groups = {"seen": test_seen, "unseen": test_unseen, "all": test_indices}
    if args.dry_run:
        print("configuration:")
        print(json.dumps(vars(args), indent=2))
        print("group_sizes:")
        print(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))
        print(f"train_size: {len(train_indices)}")
        return
    logger.info("Train size: %d", len(train_indices))
    logger.info("Group sizes: seen=%d, unseen=%d, all=%d", len(test_seen), len(test_unseen), len(test_indices))

    # Measurement A: relevance availability per group
    logger.info("=== Measurement A: Relevance availability per group (current definition) ===")
    measurement_a = {}
    for group_name, group_indices in groups.items():
        relevance = _chunked_relevance(
            dataframe, train_indices, group_indices,
            _relevance_definition_a, args.chunk_size,
        )
        stats = _measure_availability(relevance)
        measurement_a[group_name] = stats
        logger.info(
            "%s: queries=%d, all_zero=%.3f, has_rel1=%.3f, has_rel2=%.3f, has_rel3=%.3f, "
            "ceiling=%.3f",
            group_name, stats["num_queries"], stats["all_zero_fraction"],
            stats["has_rel1_fraction"], stats["has_rel2_fraction"], stats["has_rel3_fraction"],
            stats["implied_ndcg1_ceiling"],
        )
    logger.info(
        "Observed nDCG@1: seen=%.3f (ceiling %.3f), unseen=%.3f (ceiling %.3f)",
        OBSERVED_SEEN_NDCG1, measurement_a["seen"]["implied_ndcg1_ceiling"],
        OBSERVED_UNSEEN_NDCG1, measurement_a["unseen"]["implied_ndcg1_ceiling"],
    )

    # Measurement B: subtype null interaction
    logger.info("=== Measurement B: Subtype null interaction (unseen queries) ===")
    unseen_relevance = _chunked_relevance(
        dataframe, train_indices, test_unseen,
        _relevance_definition_a, args.chunk_size,
    )
    measurement_b = _measure_subtype_null(dataframe, unseen_relevance, test_unseen)
    logger.info(
        "Unseen: null_subtype=%.3f (%d queries), nonnull_has_rel1=%.3f, null_all_zero=%s",
        measurement_b["fraction_null_subtype"],
        measurement_b["num_null_subtype"],
        measurement_b["nonnull_has_rel1_fraction"],
        f"{measurement_b['null_all_zero_fraction']:.3f}" if measurement_b["null_all_zero_fraction"] is not None else "N/A",
    )

    # Measurement C: ceiling under alternative relevance definitions
    logger.info("=== Measurement C: Ceiling under alternative relevance definitions ===")
    definition_fns = {
        "A_current": _relevance_definition_a,
        "B_closure_first": _relevance_definition_b,
        "C_resolution_only": _relevance_definition_c,
    }
    measurement_c = {}
    for def_name, def_fn in definition_fns.items():
        measurement_c[def_name] = {}
        for group_name in ("seen", "unseen"):
            group_indices = groups[group_name]
            relevance = _chunked_relevance(
                dataframe, train_indices, group_indices,
                def_fn, args.chunk_size,
            )
            stats = _measure_availability(relevance)
            measurement_c[def_name][group_name] = stats
            logger.info(
                "%s %s: all_zero=%.3f, ceiling=%.3f, mean_pos_candidates=%.1f",
                def_name, group_name,
                stats["all_zero_fraction"],
                stats["implied_ndcg1_ceiling"],
                stats["mean_count_rel1"],
            )

    # Measurement D: recompute observed metrics on gradeable subset
    logger.info("=== Measurement D: Text-similarity ranker metrics on gradeable subset ===")
    text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    measurement_d = {}
    for group_name in ("seen", "unseen"):
        group_indices = groups[group_name]
        relevance = _chunked_relevance(
            dataframe, train_indices, group_indices,
            _relevance_definition_a, args.chunk_size,
        )
        similarities = _text_similarity_ranker(
            text_embeddings, group_indices, train_indices, args.chunk_size,
        )
        metrics_all = evaluate_retrieval_graded(similarities, relevance, ks=(1,))
        all_zero = np.all(relevance.toarray() == 0, axis=1)
        gradeable_mask = ~all_zero
        measurement_d[group_name] = {
            "all": {
                "nDCG@1": metrics_all["nDCG@1"],
                "MAP": metrics_all["MAP"],
                "MRR": metrics_all["MRR"],
                "num_queries": int(len(group_indices)),
            },
        }
        if np.any(gradeable_mask):
            gradeable_indices = np.flatnonzero(gradeable_mask)
            rel_gradeable = relevance[gradeable_indices]
            sim_gradeable = similarities[gradeable_indices]
            metrics_gradeable = evaluate_retrieval_graded(sim_gradeable, rel_gradeable, ks=(1,))
            measurement_d[group_name]["gradeable_only"] = {
                "nDCG@1": metrics_gradeable["nDCG@1"],
                "MAP": metrics_gradeable["MAP"],
                "MRR": metrics_gradeable["MRR"],
                "num_queries": int(np.sum(gradeable_mask)),
            }
        else:
            measurement_d[group_name]["gradeable_only"] = {
                "nDCG@1": None, "MAP": None, "MRR": None, "num_queries": 0,
            }
        logger.info(
            "%s: all nDCG@1=%.3f MAP=%.3f MRR=%.3f | gradeable nDCG@1=%s MAP=%s MRR=%s (%d of %d)",
            group_name,
            measurement_d[group_name]["all"]["nDCG@1"],
            measurement_d[group_name]["all"]["MAP"],
            measurement_d[group_name]["all"]["MRR"],
            f"{measurement_d[group_name]['gradeable_only']['nDCG@1']:.3f}" if measurement_d[group_name]["gradeable_only"]["nDCG@1"] is not None else "N/A",
            f"{measurement_d[group_name]['gradeable_only']['MAP']:.3f}" if measurement_d[group_name]["gradeable_only"]["MAP"] is not None else "N/A",
            f"{measurement_d[group_name]['gradeable_only']['MRR']:.3f}" if measurement_d[group_name]["gradeable_only"]["MRR"] is not None else "N/A",
            measurement_d[group_name]["gradeable_only"]["num_queries"],
            measurement_d[group_name]["all"]["num_queries"],
        )

    output = {
        "config": vars(args),
        "measurement_a_availability": measurement_a,
        "measurement_b_subtype_null": measurement_b,
        "measurement_c_alternative_definitions": measurement_c,
        "measurement_d_text_ranker": measurement_d,
        "observed_ndcg1": {"seen": OBSERVED_SEEN_NDCG1, "unseen": OBSERVED_UNSEEN_NDCG1},
    }
    output_path = Path("results/diagnostics/relevance_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)
    if not args.skip_plots:
        figures_dir = Path("results/diagnostics/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_ceiling(measurement_c, figures_dir / "relevance_ceiling.png")
        except Exception:
            logger.exception("Plot generation failed after JSON was written")


if __name__ == "__main__":
    main()
