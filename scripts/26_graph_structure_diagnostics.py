"""Measure relation-specific graph connectivity and purity for cold-start CIs."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
try:
    import scipy.sparse as sparse
except ImportError as error:
    raise SystemExit("scipy is required; run: uv pip install scipy") from error

from src.logging_config import setup_logging
from src.sweep_runner import (
    _prepare_dataframe,
    _set_seed,
    _split_positions,
    _training_ci_indices,
    build_graph_for_features,
)


RELATION_COLORS = {
    "focal": "#d62728",
    "train": "#1f77b4",
    "test": "#ff7f0e",
    "entity": "#7f7f7f",
}


def _forward_relations(data: object) -> list[tuple[str, str, str]]:
    """Return incident-to-entity edge types in graph order."""
    return [
        edge_type
        for edge_type in data.edge_types
        if edge_type[0] == "incident"
    ]


def _relation_matrix(data: object, edge_type: tuple[str, str, str]) -> sparse.csr_matrix:
    """Build boolean incident-by-entity CSR matrix for one graph relation."""
    source, relation, entity_type = edge_type
    edge_index = data[source, relation, entity_type].edge_index
    rows = edge_index[0].cpu().numpy()
    columns = edge_index[1].cpu().numpy()
    shape = (data["incident"].num_nodes, data[entity_type].num_nodes)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=shape, dtype=bool
    )


def _relation_info(data: object) -> dict[str, dict[str, object]]:
    """Return CSR matrices and degree metadata for every forward relation."""
    info = {}
    for edge_type in _forward_relations(data):
        relation = edge_type[1]
        matrix = _relation_matrix(data, edge_type)
        entity_degrees = np.asarray(matrix.sum(axis=0)).ravel()
        info[relation] = {
            "matrix": matrix,
            "edge_type": edge_type,
            "max_entity_degree": int(entity_degrees.max(initial=0)),
            "num_entities": matrix.shape[1],
        }
    return info


def compute_ci_degree_stats(
    data: object,
    train_incident_mask: np.ndarray,
    seen_ci_indices: object,
    unseen_ci_indices: object,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-CI edge counts split by incident split membership."""
    edge_index = data["incident", "affects", "ci"].edge_index
    incidents = edge_index[0].cpu().numpy()
    ci_nodes = edge_index[1].cpu().numpy()
    groups = {"seen": set(seen_ci_indices), "unseen": set(unseen_ci_indices)}
    output: dict[str, dict[str, np.ndarray]] = {}
    for group, ci_group in groups.items():
        totals = np.array(
            [np.count_nonzero(ci_nodes == ci_index) for ci_index in sorted(ci_group)],
            dtype=np.int64,
        )
        train_counts = np.array(
            [
                np.count_nonzero(
                    (ci_nodes == ci_index) & train_incident_mask[incidents]
                )
                for ci_index in sorted(ci_group)
            ],
            dtype=np.int64,
        )
        output[group] = {
            "total_incidents": totals,
            "train_incidents": train_counts,
            "test_incidents": totals - train_counts,
        }
    return output


def _reachable_products(
    data: object,
    test_indices: np.ndarray,
    train_incident_mask: np.ndarray,
    max_entity_degree: int,
    logger: logging.Logger,
) -> tuple[dict[str, list[sparse.csr_matrix]], dict[str, dict[str, object]]]:
    """Compute chunked boolean query-to-training products for allowed relations."""
    relation_info = _relation_info(data)
    products: dict[str, list[sparse.csr_matrix]] = {}
    skipped: dict[str, dict[str, object]] = {}
    train_indices = np.flatnonzero(train_incident_mask)
    for relation, details in relation_info.items():
        matrix = details["matrix"]
        maximum = details["max_entity_degree"]
        started = time.perf_counter()
        logger.info(
            "Starting relation %s: A shape=%s, q=%d, train=%d, max_entity_degree=%d",
            relation, matrix.shape, len(test_indices), len(train_indices), maximum,
        )
        if maximum > max_entity_degree:
            skipped[relation] = {
                "reason": "maximum entity degree exceeds limit",
                "max_entity_degree": maximum,
                "max_entity_degree_limit": max_entity_degree,
                "matrix_shape": list(matrix.shape),
            }
            logger.info(
                "Skipped relation %s: max_entity_degree=%d exceeds %d (%.2fs)",
                relation, maximum, max_entity_degree, time.perf_counter() - started,
            )
            continue
        relation_products = []
        for start in range(0, len(test_indices), 256):
            query_chunk = test_indices[start : start + 256]
            product = (matrix[query_chunk] @ matrix.T).astype(bool).tocsr()
            self_rows = np.arange(len(query_chunk), dtype=int)
            product = product.tolil()
            product[self_rows, query_chunk] = 0
            product = product.tocsr()
            relation_products.append(product[:, train_indices].tocsr())
        products[relation] = relation_products
        logger.info(
            "Finished relation %s: A shape=%s, max_entity_degree=%d, elapsed=%.2fs",
            relation, matrix.shape, maximum, time.perf_counter() - started,
        )
    return products, skipped


def _product_metrics(
    products: dict[str, list[sparse.csr_matrix]],
    test_indices: np.ndarray,
    train_incident_mask: np.ndarray,
    dataframe: object | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Derive reach counts and purity from one set of chunked products."""
    train_indices = np.flatnonzero(train_incident_mask)
    if dataframe is not None:
        codes, _ = pd.factorize(dataframe["Closure Code"], sort=True)
        closure_matrix = sparse.csr_matrix(
            (
                np.ones(len(train_indices)),
                (np.arange(len(train_indices)), codes[train_indices]),
            ),
            shape=(len(train_indices), int(codes.max()) + 1),
        )
    else:
        codes = None
        closure_matrix = None
    relation_products = dict(products)
    union_products: list[sparse.csr_matrix] = []
    for chunk_index in range((len(test_indices) + 255) // 256):
        chunk_products = [
            chunks[chunk_index]
            for chunks in products.values()
            if chunk_index < len(chunks)
        ]
        if chunk_products:
            union_product = chunk_products[0].copy()
            for relation_product in chunk_products[1:]:
                union_product += relation_product
            union_products.append((union_product > 0).tocsr())
    relation_products["union_all_relations"] = union_products
    counts = {
        relation: np.concatenate(
            [np.asarray(product.sum(axis=1)).ravel() for product in chunks]
        ).astype(np.int64)
        if chunks else np.zeros(len(test_indices), dtype=np.int64)
        for relation, chunks in relation_products.items()
    }
    purity: dict[str, np.ndarray] = {}
    if closure_matrix is None:
        return counts, purity
    for relation, chunks in relation_products.items():
        purity_chunks = []
        for chunk_index, product in enumerate(chunks):
            matching = product @ closure_matrix
            query_codes = codes[
                test_indices[chunk_index * 256 : chunk_index * 256 + product.shape[0]]
            ]
            rows = np.arange(product.shape[0])
            same_code = np.asarray(matching[rows, query_codes]).ravel()
            totals = np.asarray(product.sum(axis=1)).ravel()
            purity_chunks.append(
                np.divide(
                    same_code,
                    totals,
                    out=np.full(len(totals), np.nan),
                    where=totals != 0,
                )
            )
        purity[relation] = (
            np.concatenate(purity_chunks)
            if purity_chunks
            else np.empty(0, dtype=float)
        )
    return counts, purity


def compute_group_reach_and_purity(
    data: object,
    test_indices: np.ndarray,
    train_incident_mask: np.ndarray,
    dataframe: object,
    max_entity_degree: int = 20000,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Compute one group's reach and purity from one product pass."""
    active_logger = logger or logging.getLogger(__name__)
    products, skipped = _reachable_products(
        data, test_indices, train_incident_mask, max_entity_degree, active_logger
    )
    counts, purity = _product_metrics(
        products, test_indices, train_incident_mask, dataframe
    )
    return counts, purity, skipped


def compute_two_hop_train_reach(
    data: object,
    test_indices: np.ndarray,
    train_incident_mask: np.ndarray,
    max_entity_degree: int = 20000,
    logger: logging.Logger | None = None,
) -> dict[str, np.ndarray]:
    """Count distinct training incidents reachable within two hops per relation."""
    counts, _, _ = compute_group_reach_and_purity(
        data, test_indices, train_incident_mask, None,
        max_entity_degree, logger,
    )
    return counts


def compute_reach_purity(
    data: object,
    test_indices: np.ndarray,
    train_incident_mask: np.ndarray,
    dataframe: object,
    max_entity_degree: int = 20000,
    logger: logging.Logger | None = None,
) -> dict[str, np.ndarray]:
    """Compute closure-code purity of reachable training incidents per relation."""
    _, purity, _ = compute_group_reach_and_purity(
        data, test_indices, train_incident_mask, dataframe,
        max_entity_degree, logger,
    )
    return purity


def _sample_queries(
    indices: np.ndarray,
    max_queries: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample up to max_queries indices without replacement and sort them."""
    if len(indices) <= max_queries:
        return np.sort(indices)
    return np.sort(generator.choice(indices, size=max_queries, replace=False))


def _distribution(values: np.ndarray) -> dict[str, float]:
    """Summarize numeric values, returning zero-valued statistics when empty."""
    if len(values) == 0:
        return {key: 0.0 for key in ("mean", "median", "min", "max", "p25", "p75")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def _purity_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize purity values and count zero-reach incidents."""
    valid = values[~np.isnan(values)]
    return {
        "mean": float(np.mean(valid)) if len(valid) else float("nan"),
        "num_zero_reach": float(np.isnan(values).sum()),
    }


def _json_safe(value: object) -> object:
    """Convert NumPy values and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _relation_stats(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, dict[str, float]]:
    """Summarize relation arrays for one test group."""
    return {relation: _distribution(array[mask]) for relation, array in values.items()}


def _purity_stats(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, dict[str, float]]:
    """Summarize purity arrays for one test group."""
    return {relation: _purity_summary(array[mask]) for relation, array in values.items()}


def _select_examples(
    degree_stats: dict[str, dict[str, np.ndarray]],
    group: str,
    num_examples: int,
) -> list[int]:
    """Select CI indices nearest group median train-incident count among CIs with >=5 train incidents."""
    train_counts = degree_stats[group]["train_incidents"]
    qualified_mask = train_counts >= 5
    if not np.any(qualified_mask):
        qualified_mask = np.ones(len(train_counts), dtype=bool)
    qualified_values = train_counts[qualified_mask]
    median = float(np.median(qualified_values)) if len(qualified_values) else 0.0
    distances = np.full(len(train_counts), np.inf)
    distances[qualified_mask] = np.abs(train_counts[qualified_mask] - median)
    order = np.argsort(distances, kind="stable")[:num_examples]
    all_indices = sorted(
        degree_stats[group].get("ci_indices", [])
    )
    return [all_indices[index] for index in order]


def _add_ego_edges(graph: nx.Graph, data: object, focal: int) -> None:
    """Add all graph edges needed to represent a CI's two-hop ego network."""
    for source, relation, target in data.edge_types:
        edge_index = data[source, relation, target].edge_index
        for left, right in edge_index.t().cpu().numpy().tolist():
            left_node = (source, int(left))
            right_node = (target, int(right))
            graph.add_edge(left_node, right_node)
    graph.add_node(("ci", focal))


def _ego_nodes(graph: nx.Graph, focal: tuple[str, int], cap: int) -> nx.Graph:
    """Return a deterministic two-hop ego network capped at node count."""
    distances = nx.single_source_shortest_path_length(graph, focal, cutoff=2)
    nodes = list(distances)
    if len(nodes) > cap:
        rng = np.random.default_rng(42)
        others = [node for node in nodes if node != focal]
        nodes = [focal] + rng.choice(others, size=cap - 1, replace=False).tolist()
    return graph.subgraph(nodes).copy()


def _node_color(node: tuple[str, int], data: object, train_mask: np.ndarray, focal: int) -> str:
    """Return display color for one ego-network node."""
    node_type, index = node
    if node_type == "ci" and index == focal:
        return RELATION_COLORS["focal"]
    if node_type == "incident":
        return RELATION_COLORS["train"] if train_mask[index] else RELATION_COLORS["test"]
    return RELATION_COLORS["entity"]


def stratify_seen_by_ci_size(
    data: object,
    dataframe: object,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    buckets: list[int],
) -> dict[str, np.ndarray]:
    """Bucket seen-CI test incidents by their CI's training-incident count.

    Args:
        data: HeteroData graph.
        dataframe: Incident dataframe aligned to graph incident indices.
        train_indices: Training incident node indices.
        test_indices: Test incident node indices.
        buckets: Sorted list of right-open bucket edges over training-incident
            count, e.g. [1, 2, 5, 10, 25, 100, 1000000].

    Returns:
        Dict mapping bucket label to the array of seen test incident indices
        whose CI has a training-incident count in that bucket.
    """
    edge_index = data["incident", "affects", "ci"].edge_index
    incidents = edge_index[0].cpu().numpy()
    ci_nodes = edge_index[1].cpu().numpy()
    train_mask = np.zeros(data["incident"].num_nodes, dtype=bool)
    train_mask[train_indices] = True
    ci_train_counts = {}
    for ci_index in range(data["ci"].num_nodes):
        mask = (ci_nodes == ci_index) & train_mask[incidents]
        ci_train_counts[ci_index] = int(np.count_nonzero(mask))
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_test_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_test_mask.to_numpy()]
    test_to_ci = {}
    for incident_idx, ci_idx in zip(incidents, ci_nodes):
        if incident_idx in seen_test_indices:
            test_to_ci[incident_idx] = ci_idx
    strata = {}
    for i in range(len(buckets) - 1):
        lower = buckets[i]
        upper = buckets[i + 1]
        label = f"[{lower}, {upper})"
        indices = [
            incident_idx
            for incident_idx in seen_test_indices
            if incident_idx in test_to_ci
            and lower <= ci_train_counts[test_to_ci[incident_idx]] < upper
        ]
        strata[label] = np.array(indices, dtype=np.int64)
    return strata


def _plot_degree_distributions(degree_stats: dict, reach_stats: dict, output_dir: Path) -> None:
    """Save side-by-side log-scale CI degree histograms."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
    all_values = np.concatenate([
        degree_stats["seen"]["train_incidents"],
        degree_stats["unseen"]["train_incidents"]
    ])
    max_value = float(np.max(all_values)) if len(all_values) else 1.0
    bins = np.logspace(0, np.log10(max_value + 1), 25)
    for axis, group in zip(axes, ("seen", "unseen")):
        values = degree_stats[group]["train_incidents"]
        per_ci_median = float(np.median(values)) if len(values) else 0.0
        per_test_median = reach_stats[f"test_{group}"]["union_all_relations"]["median"]
        axis.hist(values, bins=bins, color="#4c78a8")
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_title(
            f"{group} CIs: median {per_ci_median:.0f} train incident per CI\n"
            f"median {group} test incident reaches {per_test_median:.0f}"
        )
        axis.set_xlabel("Attached train incidents")
        axis.set_ylabel("CI count")
    figure.suptitle(
        "Most CIs are small but most test incidents belong to large CIs (size-biased sampling)",
        fontsize=10, y=0.98
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "ci_degree_distributions.png", dpi=160)
    plt.close(figure)


def _plot_reachability(reach_stats: dict, relations: list[str], output_dir: Path) -> None:
    """Save grouped median reachability bars by relation."""
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(relations))
    width = 0.38
    seen = [reach_stats["test_seen"][relation]["median"] for relation in relations]
    unseen = [reach_stats["test_unseen"][relation]["median"] for relation in relations]
    seen_bars = axis.bar(positions - width / 2, seen, width, label="test_seen")
    unseen_bars = axis.bar(positions + width / 2, unseen, width, label="test_unseen")
    axis.set_xticks(positions, relations, rotation=30, ha="right")
    axis.set_ylabel("Median distinct train incidents reached")
    axis.set_yscale("symlog", linthresh=1)
    axis.set_ylim(bottom=-0.5)
    axis.bar_label(seen_bars, labels=[f"{v:.0f}" if v > 0 else "0" for v in seen], fontsize=8)
    axis.bar_label(unseen_bars, labels=[f"{v:.0f}" if v > 0 else "0" for v in unseen], fontsize=8)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "two_hop_reach_by_relation.png", dpi=160)
    plt.close(figure)


def _plot_purity(
    purity_stats: dict,
    relations: list[str],
    base_rate: float,
    output_dir: Path,
) -> None:
    """Save grouped mean purity bars by relation."""
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(relations))
    width = 0.38
    seen = [
        purity_stats["test_seen"][relation]["mean"]
        if np.isfinite(purity_stats["test_seen"][relation]["mean"])
        else 0.0
        for relation in relations
    ]
    unseen = [
        purity_stats["test_unseen"][relation]["mean"]
        if np.isfinite(purity_stats["test_unseen"][relation]["mean"])
        else 0.0
        for relation in relations
    ]
    axis.bar(positions - width / 2, seen, width, label="test_seen")
    axis.bar(positions + width / 2, unseen, width, label="test_unseen")
    axis.axhline(base_rate, color="black", linestyle="--", label=f"base rate {base_rate:.3f}")
    axis.set_xticks(positions, relations, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean reachable-set purity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reach_purity_by_relation.png", dpi=160)
    plt.close(figure)


def _plot_egos(
    data: object,
    examples: dict[str, list[int]],
    train_mask: np.ndarray,
    degree_stats: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Save representative two-hop CI ego networks with a shared legend."""
    columns = max((len(items) for items in examples.values()), default=1)
    figure, axes = plt.subplots(2, columns, figsize=(4 * columns, 8), squeeze=False)
    selection_rule = "Examples selected by median train-incident count among CIs with >=5 train incidents"
    for row, group in enumerate(("seen", "unseen")):
        for column in range(columns):
            axis = axes[row][column]
            if column >= len(examples[group]):
                axis.axis("off")
                continue
            focal = examples[group][column]
            full_graph = nx.Graph()
            _add_ego_edges(full_graph, data, focal)
            ego = _ego_nodes(full_graph, ("ci", focal), 40)
            positions = nx.spring_layout(ego, seed=42)
            colors = [_node_color(node, data, train_mask, focal) for node in ego.nodes]
            nx.draw_networkx(
                ego, positions, ax=axis, node_color=colors, node_size=35,
                with_labels=False, edge_color="#cccccc", width=0.5,
            )
            true_degree = full_graph.degree(("ci", focal))
            ci_indices = sorted(degree_stats[group]["ci_indices"])
            ci_position = ci_indices.index(focal)
            train_count = degree_stats[group]["train_incidents"][ci_position]
            test_count = degree_stats[group]["test_incidents"][ci_position]
            total_nodes = len(full_graph)
            axis.set_title(
                f"{group} CI {focal}: {train_count} train / {test_count} test incidents\n"
                f"showing {len(ego)} of {total_nodes} nodes"
            )
            axis.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label,
                   markerfacecolor=color, markersize=8)
        for label, color in (
            ("focal CI", RELATION_COLORS["focal"]),
            ("training incident", RELATION_COLORS["train"]),
            ("test incident", RELATION_COLORS["test"]),
            ("other entity", RELATION_COLORS["entity"]),
        )
    ]
    figure.suptitle(selection_rule, fontsize=10, y=0.98)
    figure.legend(handles=handles, loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    figure.savefig(output_dir / "ego_networks.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run relation-specific graph diagnostics and save metrics and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=["categorical", "text", "concat"], default="concat")
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--max-entity-degree", type=int, default=20000)
    parser.add_argument("--size-buckets", nargs="+", type=int, default=[1, 2, 5, 10, 25, 100, 1000000])
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_examples < 1 or args.max_queries < 1 or args.max_entity_degree < 1:
        raise ValueError("num-examples, max-queries, and max-entity-degree must be positive")
    logger = setup_logging("graph_structure_diagnostics")
    _set_seed(args.seed)

    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    text_embeddings = None
    if args.features in {"text", "concat"}:
        text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    data = build_graph_for_features(args.features, dataframe, text_embeddings)
    train_incident_mask = np.zeros(len(dataframe), dtype=bool)
    train_incident_mask[train_indices] = True
    seen_ci_indices = _training_ci_indices(data, train_indices)
    unseen_ci_indices = set(range(data["ci"].num_nodes)) - seen_ci_indices
    degree_stats = compute_ci_degree_stats(
        data, train_incident_mask, seen_ci_indices, unseen_ci_indices
    )
    for group, indices in (("seen", seen_ci_indices), ("unseen", unseen_ci_indices)):
        degree_stats[group]["ci_indices"] = np.asarray(sorted(indices), dtype=np.int64)

    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_test_mask = test_cis.notna() & test_cis.isin(train_cis)
    test_seen = test_indices[seen_test_mask.to_numpy()]
    test_unseen = test_indices[~seen_test_mask.to_numpy()]
    generator = np.random.default_rng(args.seed)
    test_groups = {
        "test_seen": _sample_queries(test_seen, args.max_queries, generator),
        "test_unseen": _sample_queries(test_unseen, args.max_queries, generator),
    }
    relation_info = _relation_info(data)
    skipped_relations = {
        relation: {
            "reason": "maximum entity degree exceeds limit",
            "max_entity_degree": details["max_entity_degree"],
            "max_entity_degree_limit": args.max_entity_degree,
            "matrix_shape": list(details["matrix"].shape),
        }
        for relation, details in relation_info.items()
        if details["max_entity_degree"] > args.max_entity_degree
    }
    if args.dry_run:
        print("configuration:")
        print(json.dumps(vars(args), indent=2))
        print("relations:")
        for relation, details in relation_info.items():
            matrix = details["matrix"]
            print(
                f"  {relation}: entities={details['num_entities']} "
                f"shape={matrix.shape} max_entity_degree={details['max_entity_degree']} "
                f"estimated_product_entries={len(test_groups['test_seen']) * matrix.shape[0]} "
                f"skipped={relation in skipped_relations}"
            )
        print("query_sample_sizes:")
        print(json.dumps({group: len(indices) for group, indices in test_groups.items()}, indent=2))
        return
    diagnostics = {
        group: compute_group_reach_and_purity(
            data, indices, train_incident_mask, dataframe,
            args.max_entity_degree, logger,
        )
        for group, indices in test_groups.items()
    }
    train_reach = {group: result[0] for group, result in diagnostics.items()}
    purity = {group: result[1] for group, result in diagnostics.items()}
    skipped_relations = {
        relation: details
        for result in diagnostics.values()
        for relation, details in result[2].items()
    }
    relations = list(train_reach["test_seen"].keys())
    reach_stats = {
        group: _relation_stats(train_reach[group], np.ones(len(test_groups[group]), dtype=bool))
        for group in test_groups
    }
    degree_summary = {
        group: {
            metric: _distribution(values)
            for metric, values in degree_stats[group].items()
            if metric != "ci_indices"
        }
        for group in degree_stats
    }
    purity_stats = {
        group: _purity_stats(purity[group], np.ones(len(test_groups[group]), dtype=bool))
        for group in test_groups
    }
    train_codes = dataframe.iloc[train_indices]["Closure Code"]
    base_rate = float(train_codes.value_counts(normalize=True).max())
    examples = {
        group: _select_examples(degree_stats, group, args.num_examples)
        for group in ("seen", "unseen")
    }
    size_strata = stratify_seen_by_ci_size(
        data, dataframe, train_indices, test_indices, args.size_buckets
    )
    edge_index = data["incident", "affects", "ci"].edge_index
    incidents = edge_index[0].cpu().numpy()
    ci_nodes = edge_index[1].cpu().numpy()
    test_seen_sampled = test_groups["test_seen"]
    sampled_set = set(test_seen_sampled)
    strata_reach = {}
    for label, indices in size_strata.items():
        distinct_cis = set()
        for incident_idx in indices:
            mask = incidents == incident_idx
            if np.any(mask):
                distinct_cis.add(int(ci_nodes[mask][0]))
        intersection = np.array([idx for idx in indices if idx in sampled_set], dtype=np.int64)
        if len(intersection) == 0:
            strata_reach[label] = {
                "num_test_incidents": int(len(indices)),
                "num_distinct_cis": len(distinct_cis),
                "reach_sample_n": 0,
                "median_affects_reach": None,
                "mean_affects_purity": None,
                "median_affects_purity": None,
            }
            continue
        positions = np.searchsorted(test_seen_sampled, intersection)
        affects_reach = train_reach["test_seen"]["affects"][positions]
        affects_purity = purity["test_seen"]["affects"][positions]
        median_reach = float(np.median(affects_reach))
        valid_purity = affects_purity[~np.isnan(affects_purity)]
        bucket_lower = int(label.split(",")[0].strip("["))
        bucket_upper = int(label.split(",")[1].strip(" )"))
        assert median_reach <= bucket_upper, (
            f"Bucket {label}: median affects reach {median_reach} exceeds upper bound {bucket_upper}. "
            f"This indicates wrong array indexing."
        )
        strata_reach[label] = {
            "num_test_incidents": int(len(indices)),
            "num_distinct_cis": len(distinct_cis),
            "reach_sample_n": int(len(intersection)),
            "median_affects_reach": median_reach,
            "mean_affects_purity": float(np.mean(valid_purity)) if len(valid_purity) else None,
            "median_affects_purity": float(np.median(valid_purity)) if len(valid_purity) else None,
        }
    logger.info(
        "Query samples: test_seen=%d/%d, test_unseen=%d/%d",
        len(test_groups["test_seen"]), len(test_seen),
        len(test_groups["test_unseen"]), len(test_unseen),
    )
    logger.info("Purity reference base rate (training majority closure code): %.3f", base_rate)
    logger.info("Relation          seen median/purity    unseen median/purity")
    for relation in relations:
        logger.info(
            "%-17s %.1f / %.3f          %.1f / %.3f",
            relation,
            reach_stats["test_seen"][relation]["median"],
            purity_stats["test_seen"][relation]["mean"],
            reach_stats["test_unseen"][relation]["median"],
            purity_stats["test_unseen"][relation]["mean"],
        )
    logger.info("Seen test incidents stratified by CI training-incident count:")
    logger.info("Bucket            test_incidents  distinct_cis  sample_n  median_reach  mean_purity  median_purity")
    for label in sorted(size_strata.keys()):
        stats = strata_reach[label]
        median_reach = stats["median_affects_reach"]
        mean_purity = stats["mean_affects_purity"]
        median_purity = stats["median_affects_purity"]
        logger.info(
            "%-17s %15d %13d %9d %13s %12s %14s",
            label,
            stats["num_test_incidents"],
            stats["num_distinct_cis"],
            stats["reach_sample_n"],
            f"{median_reach:.1f}" if median_reach is not None else "null",
            f"{mean_purity:.3f}" if mean_purity is not None else "null",
            f"{median_purity:.3f}" if median_purity is not None else "null",
        )
    output = {
        "config": vars(args),
        "degree_stats": degree_summary,
        "degree_counts": degree_stats,
        "reachability": reach_stats,
        "purity": purity_stats,
        "base_rate": base_rate,
        "example_ci_indices": examples,
        "query_sample_sizes": {group: len(indices) for group, indices in test_groups.items()},
        "full_population_sizes": {group: len(indices) for group, indices in (("test_seen", test_seen), ("test_unseen", test_unseen))},
        "skipped_relations": skipped_relations,
        "seen_size_strata": strata_reach,
    }
    output_path = Path("results/diagnostics/graph_structure.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)
    if not args.skip_plots:
        figures_dir = Path("results/diagnostics/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_degree_distributions(degree_stats, reach_stats, figures_dir)
            _plot_reachability(reach_stats, relations, figures_dir)
            _plot_purity(purity_stats, relations, base_rate, figures_dir)
            _plot_egos(data, examples, train_incident_mask, degree_stats, figures_dir)
        except Exception:
            logger.exception("Plot generation failed after JSON was written")


if __name__ == "__main__":
    main()
