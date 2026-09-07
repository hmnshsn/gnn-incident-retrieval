"""Train text-to-CI embedding predictors from incident profiles."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


logger = logging.getLogger(__name__)


def aggregate_ci_text_profiles(
    text_embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    data: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate training-incident text embeddings into CI-level profiles."""
    if text_embeddings.ndim != 2 or len(text_embeddings) != len(dataframe):
        raise ValueError("text_embeddings must align with dataframe rows")
    train_set = set(np.asarray(train_indices, dtype=np.int64).tolist())
    num_cis = data["ci"].num_nodes
    profiles = np.zeros((num_cis, text_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(num_cis, dtype=np.int64)
    sums = np.zeros_like(profiles)
    edge_index = data[("incident", "affects", "ci")].edge_index
    for incident_index, ci_index in edge_index.t().cpu().numpy():
        if int(incident_index) in train_set:
            sums[ci_index] += text_embeddings[incident_index]
            counts[ci_index] += 1
    nonzero = counts > 0
    profiles[nonzero] = sums[nonzero] / counts[nonzero, None]
    return profiles, counts


def aggregate_unseen_ci_text_profiles(
    text_embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    test_indices: np.ndarray,
    data: object,
    unseen_ci_indices: set[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate test-incident text embeddings for unseen CIs."""
    if text_embeddings.ndim != 2 or len(text_embeddings) != len(dataframe):
        raise ValueError("text_embeddings must align with dataframe rows")
    test_set = set(np.asarray(test_indices, dtype=np.int64).tolist())
    unseen_set = set(np.asarray(list(unseen_ci_indices), dtype=np.int64).tolist())
    num_cis = data["ci"].num_nodes
    profiles = np.zeros((num_cis, text_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(num_cis, dtype=np.int64)
    sums = np.zeros_like(profiles)
    edge_index = data[("incident", "affects", "ci")].edge_index
    for incident_index, ci_index in edge_index.t().cpu().numpy():
        if int(incident_index) in test_set and int(ci_index) in unseen_set:
            sums[ci_index] += text_embeddings[incident_index]
            counts[ci_index] += 1
    nonzero = counts > 0
    profiles[nonzero] = sums[nonzero] / counts[nonzero, None]
    return profiles, counts


def train_ci_text_predictor(
    ci_text_profiles: np.ndarray,
    ci_learned_embeddings: np.ndarray,
    seen_ci_indices: np.ndarray | list[int] | set[int],
    seed: int = 42,
) -> tuple[Ridge, dict[str, float | int]]:
    """Train Ridge regression from CI text profiles to learned embeddings."""
    np.random.seed(seed)
    if ci_text_profiles.ndim != 2 or ci_learned_embeddings.ndim != 2:
        raise ValueError("CI profiles and embeddings must be two-dimensional")
    if len(ci_text_profiles) != len(ci_learned_embeddings):
        raise ValueError("profile and embedding row counts must match")
    seen = np.asarray(list(seen_ci_indices), dtype=np.int64)
    nonzero = np.linalg.norm(ci_text_profiles[seen], axis=1) > 0
    train_indices = seen[nonzero]
    if len(train_indices) == 0:
        raise ValueError("no seen CIs have non-zero text profiles")
    predictor = Ridge(alpha=1.0)
    predictor.fit(ci_text_profiles[train_indices], ci_learned_embeddings[train_indices])
    predictions = predictor.predict(ci_text_profiles[train_indices])
    residual = predictions - ci_learned_embeddings[train_indices]
    ss_res = float(np.square(residual).sum())
    centered = ci_learned_embeddings[train_indices] - ci_learned_embeddings[train_indices].mean(axis=0)
    ss_tot = float(np.square(centered).sum())
    stats: dict[str, float | int] = {
        "num_seen_cis": int(len(seen)),
        "num_train_cis": int(len(train_indices)),
        "num_skipped_cis": int(len(seen) - len(train_indices)),
        "train_r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
        "train_mse": float(np.mean(np.square(residual))),
    }
    logger.info("CI text predictor stats: %s", stats)
    return predictor, stats


def predict_unseen_ci_embeddings(
    predictor: Ridge,
    ci_text_profiles: np.ndarray,
    unseen_ci_indices: np.ndarray | list[int] | set[int],
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Predict embeddings for unseen CIs with non-zero text profiles."""
    unseen = np.asarray(list(unseen_ci_indices), dtype=np.int64)
    has_text = np.linalg.norm(ci_text_profiles[unseen], axis=1) > 0
    predicted_indices = unseen[has_text]
    if len(predicted_indices) == 0:
        return {}, {
            "num_unseen": int(len(unseen)),
            "num_predicted": 0,
            "num_no_text": int((~has_text).sum()),
        }
    predictions = predictor.predict(ci_text_profiles[predicted_indices])
    predicted = {
        int(ci_index): embedding.astype(np.float32)
        for ci_index, embedding in zip(predicted_indices, predictions)
    }
    stats = {
        "num_unseen": int(len(unseen)),
        "num_predicted": int(len(predicted_indices)),
        "num_no_text": int((~has_text).sum()),
    }
    logger.info("CI text prediction stats: %s", stats)
    return predicted, stats
