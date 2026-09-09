"""Loss functions for resolution embedding prediction."""

import torch
import torch.nn.functional as F


def compute_resolution_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute mean cosine embedding loss between resolution embeddings.

    Args:
        predicted: Predicted embeddings with shape ``(batch_size, embed_dim)``.
        target: Target embeddings with shape ``(batch_size, embed_dim)``.

    Returns:
        Scalar mean ``1 - cosine_similarity`` loss.
    """
    predicted = F.normalize(predicted, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    cosine_sim = (predicted * target).sum(dim=1)
    return (1.0 - cosine_sim).mean()
