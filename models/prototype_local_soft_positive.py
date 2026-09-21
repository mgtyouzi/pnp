from typing import Dict, List, Sequence, Tuple

import torch

from models.prototype_reliability_rescue import PrototypeReliabilityRescue


PROTOTYPE_LOCAL_SOFT_POSITIVE_VERSION = (
    "prototype_local_soft_positive_v2_20260908"
)


class PrototypeLocalSoftPositive(PrototypeReliabilityRescue):
    """Promote one local prototype-supported candidate per matched target."""

    def __init__(
        self,
        *args,
        local_radius: float = 15.0,
        minimum_distance_improvement: float = 3.0,
        minimum_similarity_improvement: float = 0.1,
        maximum_matched_probability: float = 0.56,
        max_soft_positive_per_image: int = 8,
        soft_positive_weight: float = 0.1,
        **kwargs,
    ) -> None:
        kwargs["rescue_weight"] = 0.0
        kwargs["logit_gradient_scale"] = 0.0
        super().__init__(*args, **kwargs)
        if local_radius <= 0:
            raise ValueError("local_radius must be positive")
        if minimum_distance_improvement < 0:
            raise ValueError("minimum_distance_improvement must be non-negative")
        if minimum_similarity_improvement < 0:
            raise ValueError("minimum_similarity_improvement must be non-negative")
        if not 0 < maximum_matched_probability < 1:
            raise ValueError("maximum_matched_probability must be in (0, 1)")
        if max_soft_positive_per_image <= 0:
            raise ValueError("max_soft_positive_per_image must be positive")
        if not 0 < soft_positive_weight <= 1:
            raise ValueError("soft_positive_weight must be in (0, 1]")
        self.local_radius = float(local_radius)
        self.minimum_distance_improvement = float(
            minimum_distance_improvement
        )
        self.minimum_similarity_improvement = float(
            minimum_similarity_improvement
        )
        self.maximum_matched_probability = float(maximum_matched_probability)
        self.max_soft_positive_per_image = int(max_soft_positive_per_image)
        self.soft_positive_weight = float(soft_positive_weight)

    @torch.no_grad()
    def select_local_soft_positives(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_valid_mask: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        selected_by_image = []
        selected_probability = []
        selected_matched_probability = []
        selected_distance = []
        selected_distance_improvement = []
        selected_similarity_improvement = []
        funnel_unassigned = 0
        funnel_low_confidence_targets = 0
        funnel_local_owner = 0
        funnel_distance = 0
        funnel_similarity = 0

        prototypes = self.normalized_prototypes().detach()
        candidate_embeddings = embeddings.detach()
        probabilities = raw_logits.detach().softmax(dim=-1)[..., 0]
        active = bool(self.prototype_ready.item()) and (
            self.current_epoch >= self.warmup_epochs
        )

        for batch_index, (source_indices, target_indices) in enumerate(indices):
            device = points.device
            empty = torch.empty(0, dtype=torch.long, device=device)
            if not active:
                selected_by_image.append(empty)
                continue

            gt_points = targets["gt_points"][batch_index].to(
                device=device, dtype=torch.float32
            )
            source_indices = source_indices.to(device=device, dtype=torch.long)
            target_indices = target_indices.to(device=device, dtype=torch.long)
            valid = target_indices < int(gt_points.shape[0])
            valid = valid & (target_indices < int(support_embeddings.shape[1]))
            if support_valid_mask.numel():
                bounded_target = target_indices.clamp_max(
                    max(int(support_valid_mask.shape[1]) - 1, 0)
                )
                valid = valid & support_valid_mask[
                    batch_index, bounded_target
                ].to(device=device)
            source_indices = source_indices[valid]
            target_indices = target_indices[valid]
            if not source_indices.numel() or not gt_points.numel():
                selected_by_image.append(empty)
                continue

            matched_mask = torch.zeros(
                points.shape[1], dtype=torch.bool, device=device
            )
            matched_mask[source_indices] = True
            distances = torch.cdist(
                points[batch_index].detach().float(), gt_points.float(), p=2
            )
            nearest_distance, nearest_target = distances.min(dim=1)
            unassigned = ~matched_mask
            funnel_unassigned += int(unassigned.sum().item())

            similarities = candidate_embeddings[batch_index].matmul(
                prototypes.t()
            )
            support_similarities = support_embeddings[
                batch_index
            ].detach().matmul(prototypes.t())
            proposals = []
            for source_index, target_index in zip(
                source_indices.tolist(), target_indices.tolist()
            ):
                matched_probability = probabilities[batch_index, source_index]
                if matched_probability >= self.maximum_matched_probability:
                    continue
                funnel_low_confidence_targets += 1
                baseline_distance = distances[source_index, target_index]
                local_owner = (
                    unassigned
                    & (nearest_target == target_index)
                    & (nearest_distance <= self.local_radius)
                )
                local_indices = torch.where(local_owner)[0]
                funnel_local_owner += int(local_indices.numel())
                if not local_indices.numel():
                    continue

                distance_improvement = (
                    baseline_distance - distances[local_indices, target_index]
                )
                distance_pass = (
                    distance_improvement >= self.minimum_distance_improvement
                )
                funnel_distance += int(distance_pass.sum().item())
                local_indices = local_indices[distance_pass]
                distance_improvement = distance_improvement[distance_pass]
                if not local_indices.numel():
                    continue

                target_prototype = int(
                    support_similarities[target_index].argmax().item()
                )
                similarity_improvement = (
                    similarities[local_indices, target_prototype]
                    - similarities[source_index, target_prototype]
                )
                similarity_pass = (
                    similarity_improvement
                    >= self.minimum_similarity_improvement
                )
                funnel_similarity += int(similarity_pass.sum().item())
                local_indices = local_indices[similarity_pass]
                distance_improvement = distance_improvement[similarity_pass]
                similarity_improvement = similarity_improvement[similarity_pass]
                if not local_indices.numel():
                    continue

                utility = (
                    distance_improvement / self.local_radius
                    + similarity_improvement
                )
                best = int(utility.argmax().item())
                candidate_index = local_indices[best]
                proposals.append(
                    (
                        float(utility[best].item()),
                        candidate_index,
                        distances[candidate_index, target_index],
                        distance_improvement[best],
                        similarity_improvement[best],
                        matched_probability,
                    )
                )

            proposals.sort(key=lambda item: item[0], reverse=True)
            proposals = proposals[: self.max_soft_positive_per_image]
            if proposals:
                selected = torch.stack([item[1] for item in proposals]).long()
                selected_by_image.append(selected)
                selected_probability.append(probabilities[batch_index, selected])
                selected_matched_probability.extend(
                    item[5].reshape(1) for item in proposals
                )
                selected_distance.extend(item[2].reshape(1) for item in proposals)
                selected_distance_improvement.extend(
                    item[3].reshape(1) for item in proposals
                )
                selected_similarity_improvement.extend(
                    item[4].reshape(1) for item in proposals
                )
            else:
                selected_by_image.append(empty)

        def mean(values: List[torch.Tensor]) -> float:
            values = [value.reshape(-1).float() for value in values if value.numel()]
            return float(torch.cat(values).mean().item()) if values else float("nan")

        diagnostics = {
            "plsp_active": float(active),
            "plsp_selected": float(
                sum(int(value.numel()) for value in selected_by_image)
            ),
            "plsp_low_confidence_targets": float(funnel_low_confidence_targets),
            "plsp_unassigned": float(funnel_unassigned),
            "plsp_local_owner": float(funnel_local_owner),
            "plsp_distance_pass": float(funnel_distance),
            "plsp_similarity_pass": float(funnel_similarity),
            "plsp_selected_probability": mean(selected_probability),
            "plsp_selected_matched_probability": mean(
                selected_matched_probability
            ),
            "plsp_maximum_matched_probability": self.maximum_matched_probability,
            "plsp_selected_distance": mean(selected_distance),
            "plsp_distance_improvement": mean(
                selected_distance_improvement
            ),
            "plsp_similarity_improvement": mean(
                selected_similarity_improvement
            ),
            "plsp_positive_weight": self.soft_positive_weight,
            "plsp_regression_updates": 0.0,
            "plsp_matcher_changes": 0.0,
        }
        return selected_by_image, diagnostics
