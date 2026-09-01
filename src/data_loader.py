"""Load and clean BPI Challenge 2014 incident data.

The BPI Challenge 2014 dataset contains ITIL service-management records from
Rabobank Group ICT (incidents, changes, interactions and an incident activity
log). This module focuses on the incident-detail table, which is the primary
source for the retrieval experiments: ``Closure Code`` is the prediction
target and the remaining fields describe each incident.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default location of the raw incident-detail CSV.
DEFAULT_INCIDENT_PATH: Path = Path("data/raw/Detail_Incident.csv")

# Datetime columns present in the incident-detail table (DD/MM/YYYY HH:MM:SS).
_DATETIME_COLS: tuple[str, ...] = (
    "Open Time",
    "Reopen Time",
    "Resolved Time",
    "Close Time",
)


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing unnamed / all-empty columns from a DataFrame.

    The BPI 2014 CSVs are exported with many trailing semicolons, which pandas
    parses as unnamed, fully-empty columns.

    Args:
        df: Raw DataFrame as read by ``pd.read_csv``.

    Returns:
        DataFrame with unnamed and all-NaN columns dropped.
    """
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df = df.dropna(axis=1, how="all")
    return df


def load_bpi2014_incidents(
    path: str | Path = DEFAULT_INCIDENT_PATH,
    *,
    drop_missing_closure: bool = True,
) -> pd.DataFrame:
    """Load and clean the BPI Challenge 2014 incident-detail table.

    Reads the semicolon-delimited CSV, parses the datetime columns
    (``Open Time`` / ``Close Time`` / ...), drops trailing empty columns,
    sorts chronologically by ``Open Time`` and optionally drops rows missing
    the ``Closure Code`` target.

    Args:
        path: Path to ``Detail_Incident.csv``.
        drop_missing_closure: If True, drop rows where ``Closure Code`` is
            missing (it is the retrieval/prediction target).

    Returns:
        Cleaned DataFrame indexed sequentially and sorted by ``Open Time``.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        low_memory=False,
        dtype_backend="numpy_nullable",
    )
    df = _drop_empty_columns(df)

    # Parse datetime columns. The incident file uses DD/MM/YYYY HH:MM:SS.
    for col in _DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    if drop_missing_closure and "Closure Code" in df.columns:
        df = df.dropna(subset=["Closure Code"]).copy()

    if "Open Time" in df.columns:
        df = df.sort_values("Open Time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    *,
    time_col: str = "Open Time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted DataFrame into train / val / test by chronological order.

    The split is performed on the ``time_col`` timestamps so that all training
    incidents precede validation, which precede test (no temporal leakage).

    Args:
        df: DataFrame sorted by ``time_col`` (as produced by
            :func:`load_bpi2014_incidents`).
        train_frac: Fraction of rows for the training split.
        val_frac: Fraction of rows for the validation split. The remainder
            forms the test split.
        time_col: Column used to determine chronological order.

    Returns:
        A ``(train, val, test)`` tuple of DataFrames preserving row order.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) / [0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    if time_col not in df.columns:
        raise KeyError(f"Column {time_col!r} not found in DataFrame")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df_sorted.iloc[:train_end].reset_index(drop=True)
    val = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    test = df_sorted.iloc[val_end:].reset_index(drop=True)
    return train, val, test
