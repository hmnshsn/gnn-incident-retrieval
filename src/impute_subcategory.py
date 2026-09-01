"""Training-only CI-to-subcategory imputation utilities."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def impute_subcategory(
    df: pd.DataFrame,
    train_end_idx: int,
    confidence_threshold: float = 0.60,
    min_observations: int = 3,
) -> pd.DataFrame:
    """Impute missing subcategories using dominant training-set CI mappings.

    A CI receives a mapping only when its most frequent non-null subcategory
    accounts for at least ``confidence_threshold`` of its training observations
    and it has at least ``min_observations`` such observations. The mapping is
    then applied to missing subcategories across all rows without using
    validation or test values to construct it.

    Args:
        df: DataFrame containing ``CI Name (aff)`` and ``CI Subtype (aff)``.
        train_end_idx: Exclusive end position of the training prefix.
        confidence_threshold: Minimum dominant-subcategory proportion.
        min_observations: Minimum non-null training observations per CI.

    Returns:
        Copy of ``df`` with ``CI Subtype (aff) [imputed]`` added and
        ``CI Subtype (aff)`` filled where a confident training mapping exists.
    """
    required = {"CI Name (aff)", "CI Subtype (aff)"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    if not 0 <= train_end_idx <= len(df):
        raise ValueError("train_end_idx must be within dataframe bounds")

    result = df.copy()
    train = result.iloc[:train_end_idx]
    observed = train.dropna(subset=["CI Name (aff)", "CI Subtype (aff)"])
    counts = (
        observed.groupby(["CI Name (aff)", "CI Subtype (aff)"], dropna=True)
        .size()
        .rename("count")
        .reset_index()
    )
    if counts.empty:
        dominant = counts.assign(total=pd.Series(dtype="int64"), confidence=pd.Series(dtype="float64"))
    else:
        counts["total"] = counts.groupby("CI Name (aff)")["count"].transform("sum")
        counts["confidence"] = counts["count"] / counts["total"]
        counts["_subcategory_sort"] = counts["CI Subtype (aff)"].astype(str)
        dominant = (
            counts.sort_values(
                ["CI Name (aff)", "count", "_subcategory_sort"],
                ascending=[True, False, True],
            )
            .drop_duplicates("CI Name (aff)")
        )

    accepted = dominant[
        (dominant["total"] >= min_observations)
        & (dominant["confidence"] >= confidence_threshold)
    ]
    mapping: Mapping[object, object] = accepted.set_index("CI Name (aff)")[
        "CI Subtype (aff)"
    ].to_dict()
    imputed_values = result["CI Name (aff)"].map(mapping)
    fill_mask = result["CI Subtype (aff)"].isna() & imputed_values.notna()
    result.loc[fill_mask, "CI Subtype (aff)"] = imputed_values[fill_mask]
    result["CI Subtype (aff) [imputed]"] = result["CI Subtype (aff)"]

    print(f"Unique CIs with dominant mapping: {len(mapping)}")
    print(f"Null subcategories filled: {int(fill_mask.sum())}")
    print(
        "New subcategory null rate: "
        f"{result['CI Subtype (aff)'].isna().mean():.2%}"
    )
    print("Top 10 CI -> subcategory mappings used for imputation:")
    if accepted.empty:
        print("  none")
    else:
        for _, row in accepted.sort_values("confidence", ascending=False).head(10).iterrows():
            print(
                f"  {row['CI Name (aff)']} -> {row['CI Subtype (aff)']} "
                f"({row['confidence']:.2%})"
            )
    return result
