"""Print exploratory analysis for internal ServiceNow CSV datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "from_praison"
MAX_FULL_BYTES = 500 * 1024 * 1024
MAX_ROWS = 100_000


FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "CI-related": (
        r"(^|[_\\s])ci([_\\s]|$)",
        r"cmdb",
        r"configuration[_\\s]?item",
        r"config[_\\s]?item",
    ),
    "closure/resolution code": (
        r"closure",
        r"(^|[_\\s])close_code([_\\s]|$)",
        r"resolution[_\\s]?code",
        r"(^|[_\\s])u_resolution([_\\s]|$)",
    ),
    "category/subcategory": (r"category", r"subcategory", r"sub[_\\s]?category"),
    "assignment group": (r"assignment[_\\s]?group", r"assign[_\\s]?group"),
    "free-text description/summary": (
        r"description",
        r"summary",
        r"short[_\\s]?description",
        r"details",
        r"comments",
        r"work[_\\s]?notes",
    ),
    "timestamp": (
        r"(_at|_on|_date|_time)$",
        r"(^|_)(opened|closed|resolved|reopened|published|promoted|proposed|due)(_|$)",
    ),
    "KB article reference": (
        r"knowledge",
        r"(^|[_\\s])kb([_\\s]|$)",
        r"article",
    ),
}


ID_PATTERN = re.compile(
    r"(^|[_\s-])(id|number|no|key|sys_id|incident|article|kb)([_\s-]|$)",
    re.IGNORECASE,
)


def _read_csv(
    path: Path,
    encoding: str,
    nrows: int | None,
    separator: str | None = None,
) -> pd.DataFrame:
    """Read CSV with a Python-parser fallback for malformed quoting."""
    arguments: dict[str, Any] = {
        "encoding": encoding,
        "nrows": nrows,
        "low_memory": False,
    }
    if separator is not None:
        arguments["sep"] = separator
    try:
        return pd.read_csv(path, **arguments)
    except pd.errors.ParserError:
        print(
            f"NOTE: parser fallback for {path.name}; malformed records skipped "
            "with pandas Python engine"
        )
        arguments.pop("low_memory", None)
        arguments.update(engine="python", on_bad_lines="skip")
        return pd.read_csv(path, **arguments)


def _load_csv(path: Path) -> tuple[pd.DataFrame, bool, str]:
    """Load CSV with UTF-8 first and latin-1 fallback.

    Args:
        path: CSV path.

    Returns:
        DataFrame, truncation flag, and encoding used.
    """
    too_large = path.stat().st_size > MAX_FULL_BYTES
    nrows = MAX_ROWS if too_large else None
    encoding = "utf-8"
    try:
        dataframe = _read_csv(path, encoding, nrows)
    except UnicodeDecodeError:
        encoding = "latin-1"
        dataframe = _read_csv(path, encoding, nrows)
    if len(dataframe.columns) == 1 and ";" in str(dataframe.columns[0]):
        dataframe = _read_csv(path, encoding, nrows, separator=";")
    return dataframe, too_large, encoding


def _print_string_profiles(dataframe: pd.DataFrame) -> None:
    """Print cardinality and top five values for string-like columns."""
    string_columns = dataframe.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    if not len(string_columns):
        print("No string/object columns.")
        return
    for column in string_columns:
        values = dataframe[column].dropna()
        print(f"  {column}: nunique={values.nunique(dropna=True)}")
        print(values.value_counts(dropna=False).head(5).to_string())


def _matching_columns(dataframe: pd.DataFrame, keywords: tuple[str, ...]) -> list[str]:
    """Find columns whose lowercase names contain any keyword."""
    return [
        str(column)
        for column in dataframe.columns
        if any(re.search(keyword, str(column).lower()) for keyword in keywords)
    ]


def _print_field_checks(dataframe: pd.DataFrame) -> list[str]:
    """Print semantic column groups and return timestamp candidates."""
    timestamp_columns: list[str] = []
    for group, keywords in FIELD_GROUPS.items():
        columns = _matching_columns(dataframe, keywords)
        print(f"{group} columns: {columns or 'none found'}")
        if group == "timestamp":
            timestamp_columns = columns
        for column in columns:
            if group == "closure/resolution code":
                print(f"  {column} distribution:")
                print(dataframe[column].value_counts(dropna=False).to_string())
    return timestamp_columns


def _print_date_range(dataframe: pd.DataFrame, columns: list[str]) -> None:
    """Parse timestamp candidates and print combined minimum and maximum."""
    parsed: list[pd.Series] = []
    print("Date ranges:")
    for column in columns:
        series = pd.to_datetime(dataframe[column], errors="coerce", utc=True)
        valid = series.dropna()
        if valid.empty:
            print(f"  {column}: no parseable timestamps")
            continue
        print(f"  {column}: {valid.min()} to {valid.max()}")
        parsed.append(valid)
    if parsed:
        combined = pd.concat(parsed)
        print(f"  Combined: {combined.min()} to {combined.max()}")
    else:
        print("  No parseable timestamps found.")


def _print_dataset_analysis(name: str, path: Path) -> pd.DataFrame:
    """Print requested EDA sections for one CSV and return its DataFrame."""
    dataframe, truncated, encoding = _load_csv(path)
    print("\n" + "=" * 100)
    print(f"DATASET: {name}")
    print(f"Path: {path}")
    print(f"Encoding: {encoding}")
    print(f"Shape: {dataframe.shape}")
    if truncated:
        print(f"NOTE: file exceeds {MAX_FULL_BYTES} bytes; loaded first {MAX_ROWS:,} rows")
    else:
        print("Loaded full file.")

    print("\nColumn names:")
    for index, column in enumerate(dataframe.columns, start=1):
        print(f"  {index:>3}: {column}")
    print("\nDtypes:")
    print(dataframe.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(dataframe.head(3).T.to_string())
    print("\nNull counts:")
    print(dataframe.isna().sum().to_string())
    print("\nString/object profiles:")
    _print_string_profiles(dataframe)
    print("\nSemantic field checks:")
    timestamp_columns = _print_field_checks(dataframe)
    _print_date_range(dataframe, timestamp_columns)
    return dataframe


def _id_columns(dataframe: pd.DataFrame) -> list[str]:
    """Find likely identifier columns by name."""
    return [str(column) for column in dataframe.columns if ID_PATTERN.search(str(column))]


def _normalized_name(name: str) -> str:
    """Normalize a column name for cross-file comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _print_cross_file_analysis(
    incidents: pd.DataFrame,
    knowledge: pd.DataFrame,
) -> None:
    """Print shared-column and likely foreign-key analysis."""
    incident_columns = {str(column) for column in incidents.columns}
    knowledge_columns = {str(column) for column in knowledge.columns}
    shared = sorted(incident_columns & knowledge_columns)
    print("\n" + "=" * 100)
    print("CROSS-FILE ANALYSIS")
    print(f"Shared exact column names: {shared or 'none'}")

    incident_ids = _id_columns(incidents)
    knowledge_ids = _id_columns(knowledge)
    print(f"Incident ID-like columns: {incident_ids or 'none found'}")
    print(f"KB ID-like columns: {knowledge_ids or 'none found'}")

    normalized_incident = {
        _normalized_name(column): column for column in incident_columns
    }
    normalized_knowledge = {
        _normalized_name(column): column for column in knowledge_columns
    }
    normalized_shared = sorted(set(normalized_incident) & set(normalized_knowledge))
    print(
        "Shared normalized ID/field names: "
        f"{[(normalized_incident[key], normalized_knowledge[key]) for key in normalized_shared] or 'none'}"
    )

    candidates: list[dict[str, Any]] = []
    for incident_column in incident_ids:
        incident_values = set(incidents[incident_column].dropna().astype(str))
        if not incident_values:
            continue
        for knowledge_column in knowledge_ids:
            knowledge_values = set(knowledge[knowledge_column].dropna().astype(str))
            overlap = incident_values & knowledge_values
            if overlap:
                candidates.append(
                    {
                        "incident_column": incident_column,
                        "kb_column": knowledge_column,
                        "overlap_count": len(overlap),
                        "sample_overlap": sorted(overlap)[:5],
                    }
                )
    print(f"Foreign-key candidates with observed value overlap: {candidates or 'none found'}")


def main() -> None:
    """Run EDA for both ServiceNow datasets and print all findings."""
    incident_path = DATA_DIR / "incidents_jan2025_jan2026_combined 1.csv"
    knowledge_path = DATA_DIR / "kb_knowledge_surf 2.csv"
    incidents = _print_dataset_analysis(incident_path.name, incident_path)
    knowledge = _print_dataset_analysis(knowledge_path.name, knowledge_path)
    _print_cross_file_analysis(incidents, knowledge)


if __name__ == "__main__":
    main()
