"""Load internal ServiceNow incidents into the BPI-compatible schema."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


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

    logger.info("ServiceNow incident dataset stats:")
    logger.info("  Shape: %s", df.shape)
    logger.info("  Date range: %s to %s", df["opened_at"].min(), df["opened_at"].max())
    logger.info("  CI unique: %d", df["CI Name (aff)"].nunique(dropna=True))
    logger.info("  Resolution unique: %d", df["Closure Code"].nunique(dropna=True))
    logger.info("  Category unique: %d", df["Category"].nunique(dropna=True))
    logger.info("  Subcategory unique: %d", df["CI Subtype (aff)"].nunique(dropna=True))
    logger.info("  Null rates:")
    for column in (
        "CI Name (aff)",
        "Closure Code",
        "Category",
        "CI Subtype (aff)",
    ):
        logger.info("    %s: %.2f%%", column, df[column].isna().mean() * 100)
    return df
