"""Run and compare closure-code retrieval baselines."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.baselines import (
    category_match_baseline,
    ci_majority_baseline,
    random_baseline,
    text_similarity_baseline,
)
from src.data_loader import load_bpi2014_incidents, temporal_split
from src.evaluate import evaluate_retrieval


def _split_positions(n_rows: int, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[range, range, range]:
    """Create non-overlapping positional train, validation, and test splits."""
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    return range(train_end), range(train_end, val_end), range(val_end, n_rows)


def _format_table(results: dict[str, dict[str, float]]) -> str:
    """Format baseline metrics as a fixed-width comparison table."""
    lines = ["Method              MRR     Hits@1  Hits@3  Hits@5  Hits@10"]
    for method, metrics in results.items():
        lines.append(
            f"{method:<19}"
            f"{metrics['MRR']:.3f}   "
            f"{metrics['Hits@1']:.3f}   "
            f"{metrics['Hits@3']:.3f}   "
            f"{metrics['Hits@5']:.3f}   "
            f"{metrics['Hits@10']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Load data, run all baselines, print metrics, and save JSON results."""
    torch.manual_seed(42)
    df = load_bpi2014_incidents()
    _train_df, _val_df, test_df = temporal_split(df)
    train_indices, val_indices, test_indices = _split_positions(len(df))
    split_indices = (train_indices, val_indices, test_indices)

    n_candidates = df["Closure Code"].nunique()
    random_ranks = random_baseline(len(test_df), n_candidates, seed=42)
    category_ranks = category_match_baseline(df, split_indices)
    ci_ranks = ci_majority_baseline(df, split_indices)
    text_ranks = text_similarity_baseline(df, split_indices)

    results = {
        "Random": evaluate_retrieval(random_ranks),
        "Category-Match": evaluate_retrieval(category_ranks),
        "CI-Majority": evaluate_retrieval(ci_ranks),
        "Text-Similarity": evaluate_retrieval(text_ranks),
    }
    print(_format_table(results))

    output_path = Path("results/baselines.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
