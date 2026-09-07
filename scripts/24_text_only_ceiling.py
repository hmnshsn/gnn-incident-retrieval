"""Measure text-only retrieval ceiling for cold-start incidents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.logging_config import setup_logging
from src.sweep_runner import _evaluate_group, _prepare_dataframe, _set_seed, _split_positions



def main() -> None:
    """Evaluate cached text embeddings and write ceiling diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtype", choices=["raw", "imputed", "none"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = setup_logging("text_only_ceiling")
    _set_seed(args.seed)

    dataframe, _, graph_row_indices = _prepare_dataframe(args.subtype)
    train_indices, _, test_indices = _split_positions(len(dataframe))
    text_embeddings = np.load("results/sn_text_embeddings.npy")[graph_row_indices]
    train_cis = set(dataframe.iloc[train_indices]["CI Name (aff)"].dropna())
    test_cis = dataframe.iloc[test_indices]["CI Name (aff)"]
    seen_mask = test_cis.notna() & test_cis.isin(train_cis)
    seen_test_indices = test_indices[seen_mask.to_numpy()]
    unseen_test_indices = test_indices[~seen_mask.to_numpy()]

    results: dict[str, dict[str, float]] = {}
    for group, indices in (
        ("seen", seen_test_indices),
        ("unseen", unseen_test_indices),
        ("all", test_indices),
    ):
        if len(indices) == 0:
            continue
        results[group] = _evaluate_group(
            text_embeddings, dataframe, train_indices, indices
        )
    logger.info("Group       ndcg@1  ndcg@5  map     mrr")
    for group, metrics in results.items():
        logger.info(
            "%-10s %.3f   %.3f   %.3f   %.3f",
            group,
            metrics["ndcg@1"],
            metrics["ndcg@5"],
            metrics["map"],
            metrics["mrr"],
        )
    output = {
        "subtype": args.subtype,
        "seed": args.seed,
        "text_embedding_dim": int(text_embeddings.shape[1]),
        "num_test_seen": len(seen_test_indices),
        "num_test_unseen": len(unseen_test_indices),
        "results": results,
    }
    output_path = Path("results/diagnostics/text_only_ceiling.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("Output saved: %s", output_path)


if __name__ == "__main__":
    main()
