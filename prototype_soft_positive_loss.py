from typing import List, Optional

import torch
import torch.nn.functional as F


def cross_entropy_with_local_soft_positives(
    src_logits: torch.Tensor,
    target_classes: torch.Tensor,
    class_weight: torch.Tensor,
    soft_positive_indices: Optional[List[torch.Tensor]],
    soft_positive_weight: float,
) -> torch.Tensor:
    """Replace selected background labels with weak foreground labels."""
    if not soft_positive_indices or not any(
        int(indices.numel()) for indices in soft_positive_indices
    ):
        return F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, class_weight
        )
    if len(soft_positive_indices) != int(src_logits.shape[0]):
        raise ValueError("soft-positive index list must match batch size")
    if not 0 < float(soft_positive_weight) <= 1:
        raise ValueError("soft-positive weight must be in (0, 1]")

    background_class = int(src_logits.shape[-1] - 1)
    weak_targets = target_classes.clone()
    item_weights = src_logits.new_ones(target_classes.shape)
    for batch_index, indices in enumerate(soft_positive_indices):
        indices = indices.to(device=src_logits.device, dtype=torch.long)
        if not indices.numel():
            continue
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(
            src_logits.shape[1]
        ):
            raise IndexError("soft-positive candidate index is out of range")
        indices = torch.unique(indices)
        if not torch.all(weak_targets[batch_index, indices] == background_class):
            raise ValueError("soft positives must be unmatched background candidates")
        weak_targets[batch_index, indices] = 0
        item_weights[batch_index, indices] = float(soft_positive_weight)

    per_candidate = F.cross_entropy(
        src_logits.transpose(1, 2),
        weak_targets,
        class_weight,
        reduction="none",
    )
    original_normalizer = class_weight[target_classes].sum().clamp_min(1e-8)
    return (per_candidate * item_weights).sum() / original_normalizer
