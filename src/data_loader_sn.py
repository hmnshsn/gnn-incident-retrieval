"""Load internal ServiceNow incidents into the BPI-compatible schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SN_INCIDENT_PATH = Path(
    "data/from_praison/incidents_jan2025_jan2026_combined 1.csv"
)


def load_sn_incidents() -> pd.DataFrame:
    """Load, clean, sort, and normalize internal ServiceNow incidents.

    The loader preserves original columns while adding BPI-compatible names:
    ``CI Name (aff)``, ``CI Subtype (aff)``, ``Category``, and ``Closure Code``.
    Rows missing ``opened_at`` are removed, as are rows missing both the CI and
    resolution code. Missing resolution codes are retained for inspection and
    filtered by experiment runners before target-dependent modeling.

    Returns:
        DataFrame sorted by parsed ``opened_at`` in ascending order.
    """
    path = Path(SN_INCIDENT_PATH)
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip",
        )

    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")
    df = df.dropna(subset=["opened_at"]).copy()
    df = df.dropna(subset=["cmdb_ci", "u_resolution_code"], how="all")
    df = df.sort_values("opened_at", ascending=True).reset_index(drop=True)
    df = df.rename(
        columns={
            "cmdb_ci": "CI Name (aff)",
            "subcategory": "CI Subtype (aff)",
            "category": "Category",
            "u_resolution_code": "Closure Code",
        }
    )

    print("ServiceNow incident dataset stats:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['opened_at'].min()} to {df['opened_at'].max()}")
    print(f"  CI unique: {df['CI Name (aff)'].nunique(dropna=True)}")
    print(f"  Resolution unique: {df['Closure Code'].nunique(dropna=True)}")
    print(f"  Category unique: {df['Category'].nunique(dropna=True)}")
    print(f"  Subcategory unique: {df['CI Subtype (aff)'].nunique(dropna=True)}")
    print("  Null rates:")
    for column in (
        "CI Name (aff)",
        "Closure Code",
        "Category",
        "CI Subtype (aff)",
    ):
        print(f"    {column}: {df[column].isna().mean():.2%}")
    return df
