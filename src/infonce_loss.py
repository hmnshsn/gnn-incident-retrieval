"""Batch InfoNCE contrastive loss for incident embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_infonce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE (NT-Xent) loss over same-label positives.

    Each anchor uses all other same-label examples as positives and all other
    examples as the softmax denominator. Anchors without a positive in the
    batch are excluded from the mean.

    Args:
        embeddings: ``(batch_size, embed_dim)`` incident embeddings.
        labels: ``(batch_size,)`` integer class labels.
        temperature: Positive scaling factor for cosine similarities.

    Returns:
        Scalar mean InfoNCE loss, or a differentiable zero when no anchor has
        a positive pair in the batch.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings must be 2-D and labels must be 1-D")
    if embeddings.size(0) != labels.size(0):
        raise ValueError("embeddings and labels must have matching batch sizes")

    embeddings = F.normalize(embeddings, p=2, dim=1)
    batch_size = embeddings.size(0)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = (labels[:, None] == labels[None, :]) & not_self
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0

    logits = (embeddings @ embeddings.T) / temperature
    denominator = torch.logsumexp(logits.masked_fill(~not_self, -torch.inf), dim=1)
    positive_logsum = torch.logsumexp(
        logits.masked_fill(~positive_mask, -torch.inf), dim=1
    )
    return (denominator[valid] - positive_logsum[valid]).mean()
