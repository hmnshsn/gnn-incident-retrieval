"""Run CPU feature-level classification diagnostics for SN incidents."""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys
from typing import TextIO

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from src.data_loader_sn import load_sn_incidents


CATEGORICAL_COLUMNS = [
    "contact_type",
    "CI Subtype (aff)",
    "Category",
    "priority",
    "impact",
    "urgency",
    "business_service",
]
EMBEDDING_PATH = Path("results/sn_text_embeddings.npy")
OUTPUT_PATH = Path("results/classification_diagnostics.txt")


class Tee:
    """Write output to stdout and a file at the same time."""

    def __init__(self, stream: TextIO, file: TextIO) -> None:
        self.stream = stream
        self.file = file

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


def split_positions(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chronological train, validation, and test positions."""
    train_end = int(n_rows * 0.7)
    val_end = train_end + int(n_rows * 0.15)
    return (
        np.arange(train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n_rows),
    )


def top_k_accuracy(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Return fraction of samples whose label is in model's top-k classes."""
    probabilities = model.predict_proba(features)
    top_classes = model.classes_[
        np.argsort(probabilities, axis=1)[:, -k:]
    ]
    return float(np.mean(np.any(top_classes == labels[:, None], axis=1)))


def fit_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> LogisticRegression:
    """Fit requested logistic regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_features, train_labels)
    return model


def accuracy_lines(
    model: LogisticRegression | RandomForestClassifier,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    include_top_k: bool = False,
) -> list[str]:
    """Return train/test accuracy lines for one fitted model."""
    lines = [
        f"Train accuracy: {accuracy_score(train_labels, model.predict(train_features)):.4f}",
        f"Test accuracy: {accuracy_score(test_labels, model.predict(test_features)):.4f}",
    ]
    if include_top_k:
        lines.extend(
            [
                f"Top-3 accuracy: {top_k_accuracy(model, test_features, test_labels, 3):.4f}",
                f"Top-5 accuracy: {top_k_accuracy(model, test_features, test_labels, 5):.4f}",
            ]
        )
    return lines


def main() -> None:
    """Load data, run baselines, and tee diagnostics to stdout and file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_diagnostics()


def run_diagnostics() -> None:
    """Run all requested feature-level diagnostics."""
    dataframe = load_sn_incidents()
    original_indices = np.arange(len(dataframe), dtype=np.int64)
    valid_rows = dataframe["Closure Code"].notna()
    graph_row_indices = original_indices[valid_rows.to_numpy()]
    dataframe = dataframe.loc[valid_rows].reset_index(drop=True)

    embeddings = np.load(EMBEDDING_PATH)
    if embeddings.ndim != 2 or embeddings.shape[0] <= graph_row_indices.max():
        raise ValueError(
            "Text embeddings must have one row for every loader dataframe row"
        )
    text_features = embeddings[graph_row_indices].astype(np.float32, copy=False)

    closure_values = sorted(dataframe["Closure Code"].unique().tolist(), key=str)
    closure_mapping = {value: index for index, value in enumerate(closure_values)}
    labels = dataframe["Closure Code"].map(closure_mapping).to_numpy(dtype=np.int64)
    train_indices, _, test_indices = split_positions(len(dataframe))
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    missing_categorical = [
        column for column in CATEGORICAL_COLUMNS if column not in dataframe.columns
    ]
    categorical = dataframe.reindex(columns=CATEGORICAL_COLUMNS).fillna("MISSING").astype(str)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    categorical_features = encoder.fit_transform(categorical.iloc[train_indices])
    categorical_test = encoder.transform(categorical.iloc[test_indices])

    train_text = text_features[train_indices]
    test_text = text_features[test_indices]
    train_concat = np.hstack((train_text, categorical_features))
    test_concat = np.hstack((test_text, categorical_test))

    print("=== DATA SUMMARY ===")
    print(f"Loader rows: {len(original_indices)}")
    print(f"Filtered rows: {len(dataframe)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Aligned embedding shape: {text_features.shape}")
    print(f"Train rows: {len(train_indices)}")
    print(f"Validation rows: {len(dataframe) - len(train_indices) - len(test_indices)}")
    print(f"Test rows: {len(test_indices)}")
    print(f"Closure classes: {len(closure_values)}")
    print(f"Missing categorical columns filled with MISSING: {missing_categorical or 'none'}")

    print("\n=== CLASS DISTRIBUTION ===")
    train_counts = np.bincount(train_labels, minlength=len(closure_values))
    test_counts = np.bincount(test_labels, minlength=len(closure_values))
    order = np.argsort(-train_counts, kind="stable")
    print(f"{'Label':>5}  {'Closure Code':<35} {'Train':>7} {'Test':>7} {'Train %':>8}")
    for label in order:
        percentage = 100 * train_counts[label] / len(train_labels)
        print(
            f"{label:5d}  {str(closure_values[label]):<35} "
            f"{train_counts[label]:7d} {test_counts[label]:7d} {percentage:7.2f}%"
        )
    majority_label = int(np.argmax(train_counts))
    majority_accuracy = float(np.mean(test_labels == majority_label))
    print(
        f"Majority class baseline accuracy: {majority_accuracy:.4f} "
        f"(always predict {majority_label}: {closure_values[majority_label]})"
    )

    print("\n=== LOGISTIC REGRESSION: TEXT FEATURES ===")
    text_model = fit_logistic(train_text, train_labels)
    for line in accuracy_lines(
        text_model, train_text, test_text, train_labels, test_labels, include_top_k=True
    ):
        print(line)

    print("\n=== LOGISTIC REGRESSION: CATEGORICAL FEATURES ===")
    categorical_model = fit_logistic(categorical_features, train_labels)
    for line in accuracy_lines(
        categorical_model,
        categorical_features,
        categorical_test,
        train_labels,
        test_labels,
    ):
        print(line)

    print("\n=== LOGISTIC REGRESSION: CONCAT FEATURES ===")
    concat_model = fit_logistic(train_concat, train_labels)
    for line in accuracy_lines(
        concat_model,
        train_concat,
        test_concat,
        train_labels,
        test_labels,
        include_top_k=True,
    ):
        print(line)

    print("\n=== RANDOM FOREST: CONCAT FEATURES ===")
    forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    forest_model.fit(train_concat, train_labels)
    for line in accuracy_lines(
        forest_model, train_concat, test_concat, train_labels, test_labels
    ):
        print(line)

    test_predictions = {
        "Logistic Regression (text)": text_model.predict(test_text),
        "Logistic Regression (categorical)": categorical_model.predict(categorical_test),
        "Logistic Regression (concat)": concat_model.predict(test_concat),
        "Random Forest (concat)": forest_model.predict(test_concat),
    }
    test_accuracies = {
        name: accuracy_score(test_labels, predictions)
        for name, predictions in test_predictions.items()
    }
    best_name = max(test_accuracies, key=test_accuracies.get)
    best_predictions = test_predictions[best_name]

    print("\n=== BEST MODEL ===")
    print(f"Model: {best_name}")
    print(f"Test accuracy: {test_accuracies[best_name]:.4f}")
    print("\n=== PER-CLASS METRICS ===")
    print(
        classification_report(
            test_labels,
            best_predictions,
            labels=np.arange(len(closure_values)),
            target_names=[str(value) for value in closure_values],
            zero_division=0,
        )
    )

    print("=== TOP-10 CONFUSION PAIRS ===")
    matrix = confusion_matrix(
        test_labels, best_predictions, labels=np.arange(len(closure_values))
    )
    pairs = [
        (int(matrix[true_label, predicted_label]), true_label, predicted_label)
        for true_label in range(len(closure_values))
        for predicted_label in range(len(closure_values))
        if true_label != predicted_label and matrix[true_label, predicted_label] > 0
    ]
    for count, true_label, predicted_label in sorted(
        pairs, key=lambda item: (-item[0], item[1], item[2])
    )[:10]:
        class_total = test_counts[true_label]
        percentage = 100 * count / class_total if class_total else 0.0
        print(
            f"True: {closure_values[true_label]} -> "
            f"Pred: {closure_values[predicted_label]} | count={count} "
            f"({percentage:.2f}% of class {closure_values[true_label]})"
        )

    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
