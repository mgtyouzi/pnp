import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


PROTOTYPE_GUIDED_RANKING_VERSION = "prototype_guided_ranking_v1_20260905"


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient * ctx.scale, None


def scale_gradient(values: torch.Tensor, scale: float) -> torch.Tensor:
    return _ScaleGradient.apply(values, float(scale))


def _effective_count(shares: torch.Tensor) -> float:
    positive = shares[shares > 0]
    if positive.numel() == 0:
        return 0.0
    return float((-(positive * positive.log()).sum()).exp().item())


class PrototypeGuidedRanking(nn.Module):
    """Training-only GT prototypes that rank hard P2P candidates."""

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_prototypes: int = 4,
        warmup_epochs: int = 2,
        prototype_temperature: float = 0.1,
        ranking_temperature: float = 0.1,
        prototype_margin: float = 0.1,
        classification_margin: float = 0.2,
        prototype_rank_weight: float = 0.005,
        classification_rank_weight: float = 0.01,
        metric_weight: float = 0.005,
        center_weight: float = 0.005,
        balance_weight: float = 0.001,
        diversity_weight: float = 0.001,
        diversity_margin: float = 0.5,
        input_gradient_scale: float = 0.05,
        positive_radius: float = 15.0,
        background_radius: float = 30.0,
        hard_positive_fraction: float = 0.25,
        max_pairs_per_image: int = 32,
        max_random_background_per_image: int = 16,
        support_reservoir_size: int = 8192,
        kmeans_iterations: int = 20,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(feat_dim, hidden_dim, embedding_dim, num_prototypes) <= 0:
            raise ValueError("prototype dimensions and counts must be positive")
        if min(prototype_temperature, ranking_temperature) <= 0:
            raise ValueError("prototype temperatures must be positive")
        if not 0 <= input_gradient_scale <= 1:
            raise ValueError("input_gradient_scale must be in [0, 1]")
        if not 0 < positive_radius < background_radius:
            raise ValueError("require 0 < positive_radius < background_radius")
        if not 0 < hard_positive_fraction <= 1:
            raise ValueError("hard_positive_fraction must be in (0, 1]")
        if min(max_pairs_per_image, support_reservoir_size) <= 0:
            raise ValueError("pair and reservoir sizes must be positive")

        self.feat_dim = int(feat_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_prototypes = int(num_prototypes)
        self.warmup_epochs = int(warmup_epochs)
        self.prototype_temperature = float(prototype_temperature)
        self.ranking_temperature = float(ranking_temperature)
        self.prototype_margin = float(prototype_margin)
        self.classification_margin = float(classification_margin)
        self.prototype_rank_weight = float(prototype_rank_weight)
        self.classification_rank_weight = float(classification_rank_weight)
        self.metric_weight = float(metric_weight)
        self.center_weight = float(center_weight)
        self.balance_weight = float(balance_weight)
        self.diversity_weight = float(diversity_weight)
        self.diversity_margin = float(diversity_margin)
        self.input_gradient_scale = float(input_gradient_scale)
        self.positive_radius = float(positive_radius)
        self.background_radius = float(background_radius)
        self.hard_positive_fraction = float(hard_positive_fraction)
        self.max_pairs_per_image = int(max_pairs_per_image)
        self.max_random_background_per_image = int(
            max_random_background_per_image
        )
        self.support_reservoir_size = int(support_reservoir_size)
        self.kmeans_iterations = int(kmeans_iterations)
        self.sampling_seed = int(sampling_seed)
        self.current_epoch = 0

        self.projector = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sampling_seed + 104729)
        centers = F.normalize(
            torch.randn(num_prototypes, embedding_dim, generator=generator),
            dim=-1,
        )
        self.register_buffer("foreground_prototypes", centers)
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("sampling_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "support_reservoir",
            torch.zeros(support_reservoir_size, feat_dim),
        )
        self.register_buffer(
            "support_priorities", torch.full((support_reservoir_size,), -1.0)
        )
        self.register_buffer("support_count", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def shared_gradient_scale(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        return self.input_gradient_scale

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.foreground_prototypes.float(), dim=-1, eps=1e-6)

    def _embed(
        self, values: torch.Tensor, input_gradient_scale: float = None
    ) -> torch.Tensor:
        scale = (
            self.shared_gradient_scale()
            if input_gradient_scale is None
            else float(input_gradient_scale)
        )
        return F.normalize(
            self.projector(scale_gradient(values, scale).float()),
            dim=-1,
            eps=1e-6,
        )

    def _embed_detached(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(values.detach().float()), dim=-1, eps=1e-6)

    def foreground_scores(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        similarities = embeddings.matmul(self.normalized_prototypes().t())
        assignments = F.softmax(
            similarities / self.prototype_temperature, dim=-1
        )
        score = self.prototype_temperature * (
            torch.logsumexp(
                similarities / self.prototype_temperature, dim=-1
            )
            - math.log(self.num_prototypes)
        )
        return score, assignments

    def forward(
        self,
        cls_features: torch.Tensor,
        raw_logits: torch.Tensor,
        support_features: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self._embed(cls_features)
        if support_features is None:
            support_features = cls_features.new_zeros(
                cls_features.shape[0], 0, cls_features.shape[-1]
            )
        support_embeddings = self._embed(support_features, input_gradient_scale=0.0)
        foreground, _ = self.foreground_scores(embeddings)
        return {
            "embeddings": embeddings,
            "support_embeddings": support_embeddings,
            "candidate_raw_features": cls_features,
            "support_raw_features": support_features,
            "prototype_logits": torch.stack([foreground, -foreground], dim=-1),
            "fused_logits": raw_logits,
        }

    def _sample_without_global_rng(
        self, indices: torch.Tensor, count: int, batch_index: int
    ) -> torch.Tensor:
        if count <= 0 or indices.numel() <= count:
            return indices
        generator = torch.Generator(device="cpu")
        seed = (
            self.sampling_seed
            + 1_000_003 * int(self.sampling_step.item())
            + 9_176 * int(batch_index)
        ) % ((1 << 63) - 1)
        generator.manual_seed(seed)
        positions = torch.randperm(
            int(indices.numel()), generator=generator, device="cpu"
        )[:count].to(indices.device)
        return indices[positions]

    def _balanced_hard_positive_positions(
        self,
        source_indices: torch.Tensor,
        probability: torch.Tensor,
        embeddings: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        if count <= 0:
            return source_indices[:0]
        if not bool(self.prototype_ready.item()):
            return torch.topk(-probability[source_indices], count).indices

        _, assignments = self.foreground_scores(embeddings[source_indices])
        labels = assignments.argmax(dim=-1)
        ranked_groups = []
        for prototype_index in range(self.num_prototypes):
            positions = torch.where(labels == prototype_index)[0]
            if positions.numel():
                order = torch.argsort(probability[source_indices[positions]])
                ranked_groups.append(positions[order])
        selected = []
        depth = 0
        while len(selected) < count and ranked_groups:
            layer = [group[depth] for group in ranked_groups if depth < group.numel()]
            if not layer:
                break
            layer = torch.stack(layer)
            order = torch.argsort(probability[source_indices[layer]])
            selected.extend(layer[order].tolist())
            depth += 1
        return torch.tensor(
            selected[:count], dtype=torch.long, device=source_indices.device
        )

    def sample_candidates(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        embeddings: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        sampled = {
            "reliable_positive_indices": [],
            "reliable_positive_target_indices": [],
            "hard_positive_indices": [],
            "hard_positive_target_indices": [],
            "rejected_matched_indices": [],
            "ignored_near_indices": [],
            "eligible_background_indices": [],
            "hard_background_indices": [],
            "random_background_indices": [],
            "reliable_positive_count": [],
        }
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            device = points.device
            source_indices = source_indices.to(device=device, dtype=torch.long)
            target_indices = target_indices.to(device=device, dtype=torch.long)
            gt_points = targets["gt_points"][batch_index].to(
                device=device, dtype=torch.float32
            )
            valid_pair = target_indices < int(gt_points.shape[0])
            source_indices = source_indices[valid_pair]
            target_indices = target_indices[valid_pair]
            matched_mask = torch.zeros(
                points.shape[1], dtype=torch.bool, device=device
            )
            matched_mask[source_indices] = True

            if source_indices.numel():
                match_distance = torch.linalg.vector_norm(
                    points[batch_index, source_indices].float()
                    - gt_points[target_indices].float(),
                    dim=-1,
                )
                reliable_mask = match_distance <= self.positive_radius
                reliable_source = source_indices[reliable_mask]
                reliable_target = target_indices[reliable_mask]
                rejected_source = source_indices[~reliable_mask]
            else:
                reliable_source = source_indices
                reliable_target = target_indices
                rejected_source = source_indices

            if gt_points.numel():
                nearest = torch.cdist(
                    points[batch_index].float(), gt_points.float(), p=2
                ).min(dim=1).values
            else:
                nearest = points.new_full(
                    (points.shape[1],), float("inf"), dtype=torch.float32
                )
            ignored_near = torch.where(
                (~matched_mask) & (nearest <= self.background_radius)
            )[0]
            eligible = torch.where(
                (~matched_mask) & (nearest > self.background_radius)
            )[0]

            probability = raw_logits[batch_index].softmax(dim=-1)[:, 0]
            desired = int(math.ceil(
                int(reliable_source.numel()) * self.hard_positive_fraction
            ))
            desired = min(
                desired,
                self.max_pairs_per_image,
                int(reliable_source.numel()),
                int(eligible.numel()),
            )
            positive_positions = self._balanced_hard_positive_positions(
                reliable_source,
                probability,
                embeddings[batch_index],
                desired,
            )
            hard_positive = reliable_source[positive_positions]
            hard_positive_target = reliable_target[positive_positions]

            if desired:
                difficulty = probability[eligible]
                if bool(self.prototype_ready.item()):
                    prototype_score, _ = self.foreground_scores(
                        embeddings[batch_index, eligible]
                    )
                    normalized_score = ((prototype_score + 1.0) * 0.5).clamp(0, 1)
                    difficulty = 0.5 * difficulty + 0.5 * normalized_score
                hard_background = eligible[torch.topk(difficulty, desired).indices]
            else:
                hard_background = eligible[:0]
            hard_mask = torch.zeros(
                points.shape[1], dtype=torch.bool, device=device
            )
            hard_mask[hard_background] = True
            remaining = eligible[~hard_mask[eligible]]
            random_background = self._sample_without_global_rng(
                remaining,
                min(self.max_random_background_per_image, int(remaining.numel())),
                batch_index,
            )

            sampled["reliable_positive_indices"].append(reliable_source)
            sampled["reliable_positive_target_indices"].append(reliable_target)
            sampled["hard_positive_indices"].append(hard_positive)
            sampled["hard_positive_target_indices"].append(hard_positive_target)
            sampled["rejected_matched_indices"].append(rejected_source)
            sampled["ignored_near_indices"].append(ignored_near)
            sampled["eligible_background_indices"].append(eligible)
            sampled["hard_background_indices"].append(hard_background)
            sampled["random_background_indices"].append(random_background)
            sampled["reliable_positive_count"].append(int(reliable_source.numel()))
        self.sampling_step.add_(1)
        return sampled

    @staticmethod
    def _cat_feature_list(
        values: List[torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        values = [value for value in values if value.numel()]
        if values:
            return torch.cat(values, dim=0)
        return reference.reshape(-1, reference.shape[-1])[:0]

    def _pairwise_rank_loss(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float,
        zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not positive.numel() or not negative.numel():
            return zero, positive.new_zeros(0)
        violations = margin - positive[:, None] + negative[None, :]
        loss = self.ranking_temperature * F.softplus(
            violations / self.ranking_temperature
        ).mean()
        return loss, violations

    def _cache_raw_support(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().reshape(-1, self.feat_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.sampling_seed + 65_537 * (int(self.sampling_step.item()) + 1)
        )
        priorities = torch.rand(values.shape[0], generator=generator).to(values.device)
        count = int(self.support_count.item())
        combined_values = torch.cat([self.support_reservoir[:count], values], dim=0)
        combined_priorities = torch.cat(
            [self.support_priorities[:count], priorities], dim=0
        )
        keep = min(self.support_reservoir_size, int(combined_values.shape[0]))
        selected = torch.topk(combined_priorities, keep, sorted=False).indices
        self.support_reservoir[:keep].copy_(combined_values[selected])
        self.support_priorities[:keep].copy_(combined_priorities[selected])
        self.support_count.fill_(keep)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        candidate_raw_features: torch.Tensor,
        support_raw_features: torch.Tensor,
        support_valid_mask: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del anchor_points
        sampled = self.sample_candidates(
            points, raw_logits, embeddings.detach(), targets, indices
        )
        positive_embeddings = []
        negative_embeddings = []
        positive_raw = []
        support_raw = []
        metric_negative_raw = []
        positive_raw_margins = []
        negative_raw_margins = []

        scaled_logits = scale_gradient(raw_logits, self.shared_gradient_scale())
        scaled_margins = scaled_logits[..., 0] - scaled_logits[..., 1]
        for batch_index in range(embeddings.shape[0]):
            positive_index = sampled["hard_positive_indices"][batch_index]
            target_index = sampled["hard_positive_target_indices"][batch_index]
            hard_index = sampled["hard_background_indices"][batch_index]
            random_index = sampled["random_background_indices"][batch_index]
            valid_pair = target_index < support_raw_features.shape[1]
            if support_valid_mask.numel() and target_index.numel():
                valid_pair = valid_pair & support_valid_mask[
                    batch_index, target_index
                ]
            positive_index = positive_index[valid_pair]
            target_index = target_index[valid_pair]
            pair_count = min(int(positive_index.numel()), int(hard_index.numel()))
            positive_index = positive_index[:pair_count]
            target_index = target_index[:pair_count]
            hard_index = hard_index[:pair_count]

            positive_embeddings.append(embeddings[batch_index, positive_index])
            negative_embeddings.append(embeddings[batch_index, hard_index])
            positive_raw.append(
                candidate_raw_features[batch_index, positive_index]
            )
            support_raw.append(
                support_raw_features[batch_index, target_index]
            )
            metric_negative_raw.append(
                candidate_raw_features[
                    batch_index, torch.cat([hard_index, random_index], dim=0)
                ]
            )
            positive_raw_margins.append(
                scaled_margins[batch_index, positive_index]
            )
            negative_raw_margins.append(
                scaled_margins[batch_index, hard_index]
            )

        positive = self._cat_feature_list(positive_embeddings, embeddings)
        negative = self._cat_feature_list(negative_embeddings, embeddings)
        query_raw = self._cat_feature_list(positive_raw, candidate_raw_features)
        matched_support_raw = self._cat_feature_list(
            support_raw, support_raw_features
        )
        metric_background_raw = self._cat_feature_list(
            metric_negative_raw, candidate_raw_features
        )
        positive_margin = torch.cat(
            [value for value in positive_raw_margins if value.numel()], dim=0
        ) if any(value.numel() for value in positive_raw_margins) else raw_logits.reshape(-1)[:0]
        negative_margin = torch.cat(
            [value for value in negative_raw_margins if value.numel()], dim=0
        ) if any(value.numel() for value in negative_raw_margins) else raw_logits.reshape(-1)[:0]

        if support_valid_mask.numel():
            all_support_raw = support_raw_features[support_valid_mask.bool()]
            all_support_embeddings = support_embeddings[support_valid_mask.bool()]
        else:
            all_support_raw = support_raw_features.reshape(
                -1, support_raw_features.shape[-1]
            )[:0]
            all_support_embeddings = support_embeddings.reshape(
                -1, support_embeddings.shape[-1]
            )[:0]
        self._cache_raw_support(all_support_raw)

        zero = (
            embeddings.sum() + support_embeddings.sum() + raw_logits.sum()
        ) * 0.0
        prototype_positive = embeddings.new_zeros(0)
        prototype_negative = embeddings.new_zeros(0)
        prototype_violations = embeddings.new_zeros(0)
        cls_violations = raw_logits.new_zeros(0)
        assignments = embeddings.new_zeros(self.num_prototypes)

        if bool(self.prototype_ready.item()):
            prototype_positive, _ = self.foreground_scores(positive)
            prototype_negative, _ = self.foreground_scores(negative)
            prototype_rank_loss, prototype_violations = self._pairwise_rank_loss(
                prototype_positive,
                prototype_negative,
                self.prototype_margin,
                zero,
            )
            classification_rank_loss, cls_violations = self._pairwise_rank_loss(
                positive_margin,
                negative_margin,
                self.classification_margin,
                zero,
            )
            if all_support_embeddings.numel():
                support_scores, support_assignments = self.foreground_scores(
                    all_support_embeddings
                )
                center_loss = (1.0 - support_scores).mean()
                assignments = support_assignments.mean(dim=0)
                balance_loss = (
                    assignments
                    * (assignments.clamp_min(1e-8) * self.num_prototypes).log()
                ).sum()
                soft_centers = support_assignments.t().matmul(
                    all_support_embeddings
                ) / support_assignments.sum(dim=0).clamp_min(1e-6)[:, None]
                soft_centers = F.normalize(soft_centers, dim=-1, eps=1e-6)
            else:
                center_loss = zero
                balance_loss = zero
                soft_centers = self.normalized_prototypes()
            upper = torch.triu_indices(
                self.num_prototypes,
                self.num_prototypes,
                offset=1,
                device=soft_centers.device,
            )
            pairwise = soft_centers.matmul(soft_centers.t())
            diversity_loss = (
                F.relu(
                    pairwise[upper[0], upper[1]] - self.diversity_margin
                ).mean()
                if upper.shape[1]
                else zero
            )
        else:
            prototype_rank_loss = zero
            classification_rank_loss = zero
            center_loss = zero
            balance_loss = zero
            diversity_loss = zero

        if query_raw.numel() and metric_background_raw.numel():
            metric_query = self._embed_detached(query_raw)
            metric_support = self._embed_detached(matched_support_raw)
            metric_background = self._embed_detached(metric_background_raw)
            positive_similarity = (metric_query * metric_support).sum(dim=-1, keepdim=True)
            negative_similarity = metric_query.matmul(metric_background.t())
            metric_logits = torch.cat(
                [positive_similarity, negative_similarity], dim=1
            ) / self.prototype_temperature
            metric_loss = F.cross_entropy(
                metric_logits,
                torch.zeros(
                    metric_logits.shape[0], dtype=torch.long, device=metric_logits.device
                ),
            )
        else:
            metric_loss = zero

        loss = (
            self.prototype_rank_weight * prototype_rank_loss
            + self.classification_rank_weight * classification_rank_loss
            + self.metric_weight * metric_loss
            + self.center_weight * center_loss
            + self.balance_weight * balance_loss
            + self.diversity_weight * diversity_loss
        )
        reliable_count = sum(sampled["reliable_positive_count"])
        rejected_count = sum(
            value.numel() for value in sampled["rejected_matched_indices"]
        )
        eligible_count = sum(
            value.numel() for value in sampled["eligible_background_indices"]
        )
        hard_positive_count = sum(
            value.numel() for value in sampled["hard_positive_indices"]
        )
        hard_background_count = sum(
            value.numel() for value in sampled["hard_background_indices"]
        )
        random_background_count = sum(
            value.numel() for value in sampled["random_background_indices"]
        )
        diagnostics = {
            "pgrp_loss_raw": float(loss.detach().item()),
            "pgrp_loss_proto_rank": float(prototype_rank_loss.detach().item()),
            "pgrp_loss_cls_rank": float(classification_rank_loss.detach().item()),
            "pgrp_loss_metric": float(metric_loss.detach().item()),
            "pgrp_loss_center": float(center_loss.detach().item()),
            "pgrp_loss_balance": float(balance_loss.detach().item()),
            "pgrp_loss_diversity": float(diversity_loss.detach().item()),
            "pgrp_reliable_matched": float(reliable_count),
            "pgrp_rejected_matched": float(rejected_count),
            "pgrp_hard_positive": float(hard_positive_count),
            "pgrp_gt_support": float(all_support_raw.shape[0]),
            "pgrp_eligible_background": float(eligible_count),
            "pgrp_hard_background": float(hard_background_count),
            "pgrp_random_background": float(random_background_count),
            "pgrp_ignored_near": float(sum(
                value.numel() for value in sampled["ignored_near_indices"]
            )),
            "pgrp_effective_prototypes": _effective_count(assignments.detach()),
            "pgrp_assignment_share_min": float(assignments.min().item())
            if assignments.numel() else 0.0,
            "pgrp_shared_gradient_scale": self.shared_gradient_scale(),
            "pgrp_support_cached": float(self.support_count.item()),
            "prototype_ready": float(self.prototype_ready.item()),
        }
        if positive_margin.numel() and negative_margin.numel():
            raw_positive = float(positive_margin.detach().mean().item())
            raw_background = float(negative_margin.detach().mean().item())
            diagnostics.update({
                "pgrp_raw_positive": raw_positive,
                "pgrp_raw_background": raw_background,
                "pgrp_cls_rank_gap": raw_positive - raw_background,
                "pgrp_cls_violation_rate": float(
                    (cls_violations.detach() > 0).float().mean().item()
                ),
            })
        if prototype_positive.numel() and prototype_negative.numel():
            proto_positive = float(prototype_positive.detach().mean().item())
            proto_background = float(prototype_negative.detach().mean().item())
            diagnostics.update({
                "pgrp_proto_positive": proto_positive,
                "pgrp_proto_background": proto_background,
                "pgrp_proto_rank_gap": proto_positive - proto_background,
                "pgrp_proto_violation_rate": float(
                    (prototype_violations.detach() > 0).float().mean().item()
                ),
            })
        return loss, diagnostics

    def begin_epoch(self) -> None:
        self.support_count.zero_()
        self.support_priorities.fill_(-1.0)

    def maybe_refresh_prototypes(self) -> Dict[str, float]:
        return {}

    def _spherical_kmeans(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        if values.shape[0] < self.num_prototypes:
            raise RuntimeError("not enough exact-GT supports to initialize prototypes")
        first = self.sampling_seed % int(values.shape[0])
        centers = [values[first]]
        for _ in range(1, self.num_prototypes):
            existing = torch.stack(centers, dim=0)
            nearest = values.matmul(existing.t()).max(dim=1).values
            centers.append(values[nearest.argmin()])
        centers = torch.stack(centers, dim=0)
        for _ in range(self.kmeans_iterations):
            assignment = values.matmul(centers.t()).argmax(dim=1)
            updated = []
            for prototype_index in range(self.num_prototypes):
                members = values[assignment == prototype_index]
                updated.append(
                    F.normalize(members.mean(dim=0), dim=0, eps=1e-6)
                    if members.numel()
                    else centers[prototype_index]
                )
            next_centers = torch.stack(updated, dim=0)
            if torch.allclose(next_centers, centers, atol=1e-5, rtol=0.0):
                centers = next_centers
                break
            centers = next_centers
        return centers

    def _project_raw_support(self, values: torch.Tensor) -> torch.Tensor:
        device = next(self.projector.parameters()).device
        chunks = []
        with torch.no_grad():
            for start in range(0, int(values.shape[0]), 4096):
                chunks.append(
                    self._embed_detached(values[start:start + 4096].to(device))
                )
        return torch.cat(chunks, dim=0)

    def finalize_epoch(self) -> Dict[str, float]:
        local = self.support_reservoir[
            : int(self.support_count.item())
        ].detach().cpu()
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local)
            rank = dist.get_rank()
        else:
            gathered = [local]
            rank = 0
        event = 0.0
        if rank == 0:
            values = torch.cat(gathered, dim=0)
            if values.shape[0] >= self.num_prototypes:
                projected = self._project_raw_support(values)
                centers = self._spherical_kmeans(projected).to(
                    self.foreground_prototypes.device
                )
                event = 1.0
            else:
                centers = self.foreground_prototypes.detach().clone()
        else:
            centers = torch.zeros_like(self.foreground_prototypes)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(centers, src=0)
            event_tensor = torch.tensor(
                event, device=self.foreground_prototypes.device
            )
            dist.broadcast(event_tensor, src=0)
            event = float(event_tensor.item())
        was_ready = bool(self.prototype_ready.item())
        if event:
            self.foreground_prototypes.copy_(centers)
            self.prototype_ready.fill_(1)

        prototypes = self.normalized_prototypes().detach()
        pairwise = prototypes.matmul(prototypes.t())
        if self.num_prototypes > 1:
            pairwise.fill_diagonal_(-1.0)
            pairwise_max = float(pairwise.max().item())
        else:
            pairwise_max = float("nan")
        return {
            "pgrp_initialization_event": float(event and not was_ready),
            "pgrp_refresh_event": float(event and was_ready),
            "pgrp_foreground_pairwise_similarity_max": pairwise_max,
            "pgrp_shared_gradient_scale": self.shared_gradient_scale(),
            "pgrp_raw_support_gathered": float(sum(x.shape[0] for x in gathered)),
            "pgrp_inference_fusion": 0.0,
        }
