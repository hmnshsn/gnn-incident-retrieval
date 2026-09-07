"""Batch triplet loss with semi-hard negative mining."""

import torch
import torch.nn.functional as F


def batch_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, int]:
    """Compute triplet margin loss with semi-hard mining on a batch of embeddings.

    Uses cosine distance on L2-normalized embeddings. For each anchor, selects
    the hardest positive and closest semi-hard negative, falling back to the
    closest negative when no semi-hard negative exists.

    Args:
        embeddings: ``(N, D)`` incident embeddings.
        labels: ``(N,)`` integer closure-code class indices.
        margin: Triplet margin in cosine-distance space.

    Returns:
        Scalar mean loss over valid anchors and number of active triplets.
    """
    emb = F.normalize(embeddings, p=2, dim=1)
    sim = emb @ emb.T
    dist = 1.0 - sim
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    label_not_equal = ~label_equal
    not_self = ~torch.eye(
        embeddings.size(0), dtype=torch.bool, device=embeddings.device
    )
    positive_mask = label_equal & not_self
    negative_mask = label_not_equal
    has_positive = positive_mask.any(dim=1)
    has_negative = negative_mask.any(dim=1)

    positive_dist = dist * positive_mask.float()
    d_ap = positive_dist.max(dim=1).values

    neg_dist = dist.clone()
    neg_dist[~negative_mask] = float("inf")
    d_an_hard = neg_dist.min(dim=1).values

    semi_hard = (
        negative_mask
        & (dist > d_ap.unsqueeze(1))
        & (dist < d_ap.unsqueeze(1) + margin)
    )
    semi_hard_dist = dist.clone()
    semi_hard_dist[~semi_hard] = float("inf")
    d_an_semi = semi_hard_dist.min(dim=1).values
    d_an = torch.where(semi_hard.any(dim=1), d_an_semi, d_an_hard)

    loss_per_anchor = F.relu(d_ap - d_an + margin)
    valid = has_positive & has_negative
    if not valid.any():
        return torch.tensor(
            0.0,
            device=embeddings.device,
            dtype=embeddings.dtype,
            requires_grad=True,
        ), 0
    valid_loss = loss_per_anchor[valid]
    return valid_loss.mean(), int((valid_loss > 0).sum().item())
