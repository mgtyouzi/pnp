import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION = (
    "source_supervised_candidate_proto_v1_20260831"
)


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
    entropy = -(positive * positive.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def _pairwise_max(values: torch.Tensor) -> float:
    if values.shape[0] < 2:
        return float("nan")
    matrix = values.matmul(values.t())
    matrix.fill_diagonal_(-1.0)
    return float(matrix.max().item())


class SourceSupervisedCandidatePrototype(nn.Module):
    """Supervised candidate prototypes operating on native P2P cls features."""

    def __init__(
        self,
        feat_dim: int = 256,
        num_foreground_prototypes: int = 4,
        num_background_prototypes: int = 8,
        foreground_queue_size: int = 8192,
        background_queue_size: int = 16384,
        max_hard_positive_per_image: int = 32,
        max_random_positive_per_image: int = 32,
        max_hard_background_per_image: int = 32,
        max_random_background_per_image: int = 32,
        positive_radius: float = 15.0,
        background_radius: float = 30.0,
        prototype_temperature: float = 0.1,
        prototype_margin: float = 0.1,
        prototype_momentum: float = 0.95,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        minimum_assignment_share: float = 0.02,
        dead_prototype_patience: int = 2,
        alpha_max: float = 0.25,
        alpha_initial: float = 0.001,
        confidence_threshold: float = 0.05,
        fusion_clip: float = 0.5,
        proto_ce_weight: float = 0.02,
        margin_weight: float = 0.02,
        fused_ce_weight: float = 0.05,
        input_gradient_scale: float = 0.05,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(feat_dim, num_foreground_prototypes, num_background_prototypes) < 1:
            raise ValueError("feature dimension and prototype counts must be positive")
        if foreground_queue_size < num_foreground_prototypes:
            raise ValueError("foreground queue is smaller than its prototype bank")
        if background_queue_size < num_background_prototypes:
            raise ValueError("background queue is smaller than its prototype bank")
        if positive_radius <= 0 or background_radius <= positive_radius:
            raise ValueError("background_radius must exceed positive_radius")
        if prototype_temperature <= 0 or sinkhorn_epsilon <= 0:
            raise ValueError("temperatures must be positive")
        if not 0 <= prototype_momentum < 1:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if not 0 <= minimum_assignment_share < 1:
            raise ValueError("minimum_assignment_share must be in [0, 1)")
        if not 0 < alpha_max <= 1 or not 0 <= alpha_initial < alpha_max:
            raise ValueError("alpha_initial must be in [0, alpha_max)")
        if fusion_clip <= 0:
            raise ValueError("fusion_clip must be positive")

        self.feat_dim = int(feat_dim)
        self.num_fg = int(num_foreground_prototypes)
        self.num_bg = int(num_background_prototypes)
        self.foreground_queue_size = int(foreground_queue_size)
        self.background_queue_size = int(background_queue_size)
        self.max_hard_positive_per_image = int(max_hard_positive_per_image)
        self.max_random_positive_per_image = int(max_random_positive_per_image)
        self.max_hard_background_per_image = int(max_hard_background_per_image)
        self.max_random_background_per_image = int(max_random_background_per_image)
        self.positive_radius = float(positive_radius)
        self.background_radius = float(background_radius)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_margin = float(prototype_margin)
        self.prototype_momentum = float(prototype_momentum)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.minimum_assignment_share = float(minimum_assignment_share)
        self.dead_prototype_patience = int(dead_prototype_patience)
        self.alpha_max = float(alpha_max)
        self.confidence_threshold = float(confidence_threshold)
        self.fusion_clip = float(fusion_clip)
        self.proto_ce_weight = float(proto_ce_weight)
        self.margin_weight = float(margin_weight)
        self.fused_ce_weight = float(fused_ce_weight)
        self.input_gradient_scale = float(input_gradient_scale)
        self.sampling_seed = int(sampling_seed)

        alpha_ratio = max(alpha_initial / alpha_max, 1e-6)
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        self.alpha_foreground_logit = nn.Parameter(torch.tensor(alpha_logit))
        self.alpha_background_logit = nn.Parameter(torch.tensor(alpha_logit))

        self.register_buffer(
            "foreground_prototypes", torch.zeros(self.num_fg, self.feat_dim)
        )
        self.register_buffer(
            "background_prototypes", torch.zeros(self.num_bg, self.feat_dim)
        )
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer(
            "foreground_queue", torch.zeros(self.foreground_queue_size, self.feat_dim)
        )
        self.register_buffer(
            "background_queue", torch.zeros(self.background_queue_size, self.feat_dim)
        )
        self.register_buffer(
            "foreground_queue_priorities",
            torch.full((self.foreground_queue_size,), -1.0, dtype=torch.float64),
        )
        self.register_buffer(
            "background_queue_priorities",
            torch.full((self.background_queue_size,), -1.0, dtype=torch.float64),
        )
        self.register_buffer("foreground_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("background_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("foreground_seen_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("background_seen_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("foreground_dead_age", torch.zeros(self.num_fg, dtype=torch.long))
        self.register_buffer("background_dead_age", torch.zeros(self.num_bg, dtype=torch.long))
        self.register_buffer("sampling_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prototype_update_count", torch.tensor(0, dtype=torch.long))
        self.bank_id = ""

    def normalized_foreground_prototypes(self) -> torch.Tensor:
        return F.normalize(self.foreground_prototypes.float(), dim=-1, eps=1e-6)

    def normalized_background_prototypes(self) -> torch.Tensor:
        return F.normalize(self.background_prototypes.float(), dim=-1, eps=1e-6)

    def current_alphas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        foreground = self.alpha_max * torch.sigmoid(self.alpha_foreground_logit)
        background = self.alpha_max * torch.sigmoid(self.alpha_background_logit)
        return foreground, background

    def class_scores(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        foreground_similarity = embeddings.matmul(
            self.normalized_foreground_prototypes().t()
        )
        background_similarity = embeddings.matmul(
            self.normalized_background_prototypes().t()
        )
        temperature = self.prototype_temperature
        foreground = temperature * (
            torch.logsumexp(foreground_similarity / temperature, dim=-1)
            - math.log(self.num_fg)
        )
        background = temperature * (
            torch.logsumexp(background_similarity / temperature, dim=-1)
            - math.log(self.num_bg)
        )
        return foreground, background

    def forward(self, cls_features: torch.Tensor, raw_logits: torch.Tensor) -> Dict:
        scaled_features = scale_gradient(cls_features, self.input_gradient_scale)
        embeddings = F.normalize(scaled_features.float(), dim=-1, eps=1e-6)
        if not bool(self.prototype_ready.item()):
            zeros = raw_logits.new_zeros(raw_logits.shape[:-1])
            return {
                "embeddings": embeddings,
                "prototype_logits": torch.stack([zeros, zeros], dim=-1),
                "prototype_distances": torch.stack(
                    [torch.ones_like(zeros), torch.ones_like(zeros)], dim=-1
                ),
                "prototype_margin": zeros,
                "fusion_delta": zeros,
                "fused_logits": raw_logits,
            }

        foreground, background = self.class_scores(embeddings)
        margin = foreground - background
        alpha_foreground, alpha_background = self.current_alphas()
        delta = (
            alpha_foreground * F.relu(margin - self.confidence_threshold)
            - alpha_background * F.relu(-margin - self.confidence_threshold)
        ).clamp(-self.fusion_clip, self.fusion_clip)
        # The original detector loss owns raw logits. The auxiliary fused CE
        # must not create a second, unscaled gradient path into the P2P head.
        fused_logits = raw_logits.detach().float().clone()
        fused_logits[..., 0] = fused_logits[..., 0] + 0.5 * delta
        fused_logits[..., 1] = fused_logits[..., 1] - 0.5 * delta
        return {
            "embeddings": embeddings,
            "prototype_logits": torch.stack([foreground, background], dim=-1),
            "prototype_distances": torch.stack(
                [1.0 - foreground, 1.0 - background], dim=-1
            ),
            "prototype_margin": margin,
            "fusion_delta": delta,
            "fused_logits": fused_logits.to(dtype=raw_logits.dtype),
        }

    def load_bank_payload(self, payload: Dict) -> Dict[str, object]:
        if payload.get("implementation_version") != SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION:
            raise RuntimeError("source-supervised prototype bank version mismatch")
        foreground = torch.as_tensor(payload.get("foreground_prototypes"))
        background = torch.as_tensor(payload.get("background_prototypes"))
        if tuple(foreground.shape) != tuple(self.foreground_prototypes.shape):
            raise RuntimeError("foreground prototype shape mismatch")
        if tuple(background.shape) != tuple(self.background_prototypes.shape):
            raise RuntimeError("background prototype shape mismatch")
        with torch.no_grad():
            self.foreground_prototypes.copy_(
                F.normalize(foreground.float(), dim=-1, eps=1e-6)
            )
            self.background_prototypes.copy_(
                F.normalize(background.float(), dim=-1, eps=1e-6)
            )
            self.prototype_ready.fill_(1)
            self.sampling_step.zero_()
        self.bank_id = str(payload.get("bank_id", ""))
        metadata = dict(payload.get("metadata", {}))
        metadata.update({
            "mode": "source_supervised_candidate_proto",
            "implementation_version": SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION,
            "bank_id": self.bank_id,
            "inference_fusion": True,
        })
        return metadata

    def load_bank_file(self, path: str) -> Dict[str, object]:
        payload = torch.load(str(path), map_location="cpu")
        return self.load_bank_payload(payload)

    def sample_candidates(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        sampled = {
            "positive_indices": [],
            "hard_positive_indices": [],
            "random_positive_indices": [],
            "rejected_match_indices": [],
            "ignored_near_indices": [],
            "background_indices": [],
            "hard_background_indices": [],
            "random_background_indices": [],
        }
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            device = points.device
            source_indices = source_indices.to(device=device, dtype=torch.long)
            target_indices = target_indices.to(device=device, dtype=torch.long)
            gt_points = targets["gt_points"][batch_index].to(
                device=device, dtype=torch.float32
            )
            matched_mask = torch.zeros(points.shape[1], dtype=torch.bool, device=device)
            matched_mask[source_indices] = True
            cell_probability = raw_logits[batch_index].softmax(dim=-1)[:, 0]

            if source_indices.numel():
                distances = torch.norm(
                    points[batch_index, source_indices].float()
                    - gt_points[target_indices],
                    dim=-1,
                )
                reliable = source_indices[distances <= self.positive_radius]
                rejected = source_indices[distances > self.positive_radius]
            else:
                reliable = source_indices
                rejected = source_indices

            hard_positive_count = min(
                self.max_hard_positive_per_image, int(reliable.numel())
            )
            if hard_positive_count:
                hard_positive = reliable[
                    torch.topk(
                        -cell_probability[reliable], hard_positive_count, sorted=False
                    ).indices
                ]
            else:
                hard_positive = reliable[:0]
            positive_mask = torch.zeros(points.shape[1], dtype=torch.bool, device=device)
            positive_mask[hard_positive] = True
            remaining_positive = reliable[~positive_mask[reliable]]
            random_positive = self._sample_without_global_rng(
                remaining_positive,
                min(self.max_random_positive_per_image, int(remaining_positive.numel())),
                batch_index,
                class_offset=17,
            )
            selected_positive = torch.cat([hard_positive, random_positive], dim=0)

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
            eligible_background = torch.where(
                (~matched_mask) & (nearest > self.background_radius)
            )[0]
            hard_background_count = min(
                self.max_hard_background_per_image,
                int(eligible_background.numel()),
            )
            if hard_background_count:
                hard_background = eligible_background[
                    torch.topk(
                        cell_probability[eligible_background],
                        hard_background_count,
                        sorted=False,
                    ).indices
                ]
            else:
                hard_background = eligible_background[:0]
            background_mask = torch.zeros(points.shape[1], dtype=torch.bool, device=device)
            background_mask[hard_background] = True
            remaining_background = eligible_background[
                ~background_mask[eligible_background]
            ]
            random_background = self._sample_without_global_rng(
                remaining_background,
                min(
                    self.max_random_background_per_image,
                    int(remaining_background.numel()),
                ),
                batch_index,
                class_offset=31,
            )
            selected_background = torch.cat(
                [hard_background, random_background], dim=0
            )

            sampled["positive_indices"].append(selected_positive)
            sampled["hard_positive_indices"].append(hard_positive)
            sampled["random_positive_indices"].append(random_positive)
            sampled["rejected_match_indices"].append(rejected)
            sampled["ignored_near_indices"].append(ignored_near)
            sampled["background_indices"].append(selected_background)
            sampled["hard_background_indices"].append(hard_background)
            sampled["random_background_indices"].append(random_background)
        self.sampling_step.add_(1)
        return sampled

    def _sample_without_global_rng(
        self,
        indices: torch.Tensor,
        count: int,
        batch_index: int,
        class_offset: int,
    ) -> torch.Tensor:
        if count <= 0 or indices.numel() == 0:
            return indices[:0]
        if indices.numel() <= count:
            return indices
        seed = (
            self.sampling_seed
            + 1_000_003 * int(self.sampling_step.item())
            + 9_176 * int(batch_index)
            + int(class_offset)
        ) % ((1 << 63) - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        positions = torch.randperm(
            int(indices.numel()), generator=generator, device="cpu"
        )[:count].to(indices.device)
        return indices[positions]

    def _balanced_values(
        self, foreground: torch.Tensor, background: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        count = min(int(foreground.shape[0]), int(background.shape[0]))
        if count <= 0:
            return foreground[:0], background[:0], 0
        foreground_positions = torch.linspace(
            0, foreground.shape[0] - 1, count, device=foreground.device
        ).round().long()
        background_positions = torch.linspace(
            0, background.shape[0] - 1, count, device=background.device
        ).round().long()
        return foreground[foreground_positions], background[background_positions], count

    def candidate_losses(
        self,
        positive_embeddings: torch.Tensor,
        background_embeddings: torch.Tensor,
        positive_fused_logits: Optional[torch.Tensor],
        background_fused_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        positive, background, count = self._balanced_values(
            positive_embeddings, background_embeddings
        )
        reference = positive_embeddings if positive_embeddings.numel() else background_embeddings
        zero = reference.sum() * 0.0
        if not bool(self.prototype_ready.item()) or count == 0:
            return zero, {
                "sscp_loss_proto_ce": 0.0,
                "sscp_loss_margin": 0.0,
                "sscp_loss_fused_ce": 0.0,
                "sscp_balanced_count": count,
            }

        positive_fg, positive_bg = self.class_scores(positive)
        background_fg, background_bg = self.class_scores(background)
        positive_proto_logits = torch.stack([positive_fg, positive_bg], dim=-1)
        background_proto_logits = torch.stack([background_fg, background_bg], dim=-1)
        proto_ce = 0.5 * (
            F.cross_entropy(
                positive_proto_logits,
                torch.zeros(count, dtype=torch.long, device=positive.device),
            )
            + F.cross_entropy(
                background_proto_logits,
                torch.ones(count, dtype=torch.long, device=background.device),
            )
        )
        positive_margin = positive_fg - positive_bg
        background_margin = background_fg - background_bg
        margin_loss = 0.5 * (
            F.relu(self.prototype_margin - positive_margin).mean()
            + F.relu(self.prototype_margin + background_margin).mean()
        )

        fused_ce = zero
        if positive_fused_logits is not None and background_fused_logits is not None:
            positive_fused, background_fused, fused_count = self._balanced_values(
                positive_fused_logits, background_fused_logits
            )
            if fused_count:
                fused_ce = 0.5 * (
                    F.cross_entropy(
                        positive_fused,
                        torch.zeros(
                            fused_count, dtype=torch.long, device=positive_fused.device
                        ),
                    )
                    + F.cross_entropy(
                        background_fused,
                        torch.ones(
                            fused_count, dtype=torch.long, device=background_fused.device
                        ),
                    )
                )

        loss = (
            self.proto_ce_weight * proto_ce
            + self.margin_weight * margin_loss
            + self.fused_ce_weight * fused_ce
        )
        return loss, {
            "sscp_loss_proto_ce": float(proto_ce.detach().item()),
            "sscp_loss_margin": float(margin_loss.detach().item()),
            "sscp_loss_fused_ce": float(fused_ce.detach().item()),
            "sscp_loss_proto_ce_weighted": float(
                (self.proto_ce_weight * proto_ce).detach().item()
            ),
            "sscp_loss_margin_weighted": float(
                (self.margin_weight * margin_loss).detach().item()
            ),
            "sscp_loss_fused_ce_weighted": float(
                (self.fused_ce_weight * fused_ce).detach().item()
            ),
            "sscp_positive_margin": float(positive_margin.detach().mean().item()),
            "sscp_background_margin": float(background_margin.detach().mean().item()),
            "sscp_balanced_count": count,
        }

    def compute_loss_and_cache(
        self,
        embeddings: torch.Tensor,
        fused_logits: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        **_: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sampled = self.sample_candidates(points, raw_logits, targets, indices)
        positive_embeddings = []
        background_embeddings = []
        positive_fused_logits = []
        background_fused_logits = []
        for batch_index in range(embeddings.shape[0]):
            positive_indices = sampled["positive_indices"][batch_index]
            background_indices = sampled["background_indices"][batch_index]
            if positive_indices.numel():
                positive_embeddings.append(embeddings[batch_index, positive_indices])
                positive_fused_logits.append(fused_logits[batch_index, positive_indices])
            if background_indices.numel():
                background_embeddings.append(embeddings[batch_index, background_indices])
                background_fused_logits.append(
                    fused_logits[batch_index, background_indices]
                )
        positive = (
            torch.cat(positive_embeddings, dim=0)
            if positive_embeddings
            else embeddings.new_zeros(0, self.feat_dim)
        )
        background = (
            torch.cat(background_embeddings, dim=0)
            if background_embeddings
            else embeddings.new_zeros(0, self.feat_dim)
        )
        positive_fused = (
            torch.cat(positive_fused_logits, dim=0)
            if positive_fused_logits
            else fused_logits.new_zeros(0, 2)
        )
        background_fused = (
            torch.cat(background_fused_logits, dim=0)
            if background_fused_logits
            else fused_logits.new_zeros(0, 2)
        )
        self._cache_foreground(positive.detach())
        self._cache_background(background.detach())
        loss, diagnostics = self.candidate_losses(
            positive, background, positive_fused, background_fused
        )
        alpha_foreground, alpha_background = self.current_alphas()
        diagnostics.update({
            "prototype_ready": float(self.prototype_ready.item()),
            "sscp_loss_raw": float(loss.detach().item()),
            "sscp_hard_positive": sum(
                int(value.numel()) for value in sampled["hard_positive_indices"]
            ),
            "sscp_random_positive": sum(
                int(value.numel()) for value in sampled["random_positive_indices"]
            ),
            "sscp_rejected_match": sum(
                int(value.numel()) for value in sampled["rejected_match_indices"]
            ),
            "sscp_ignored_near": sum(
                int(value.numel()) for value in sampled["ignored_near_indices"]
            ),
            "sscp_hard_background": sum(
                int(value.numel()) for value in sampled["hard_background_indices"]
            ),
            "sscp_random_background": sum(
                int(value.numel()) for value in sampled["random_background_indices"]
            ),
            "sscp_alpha_foreground": float(alpha_foreground.detach().item()),
            "sscp_alpha_background": float(alpha_background.detach().item()),
        })
        return loss, diagnostics

    def _priorities(self, count: int, offset: int, device: torch.device) -> torch.Tensor:
        indices = torch.arange(count, dtype=torch.long, device=device)
        hashed = torch.bitwise_and(
            (indices + offset) * 1103515245 + (self.sampling_seed + 1) * 12345,
            torch.tensor(0x7FFFFFFF, dtype=torch.long, device=device),
        )
        return hashed.to(torch.float64) / float(0x7FFFFFFF)

    def _cache(
        self,
        values: torch.Tensor,
        queue: torch.Tensor,
        priorities: torch.Tensor,
        count_buffer: torch.Tensor,
        seen_buffer: torch.Tensor,
    ) -> None:
        if values.numel() == 0:
            return
        values = F.normalize(values.detach().float(), dim=-1, eps=1e-6)
        old_count = int(count_buffer.item())
        old_values = queue[:old_count]
        old_priorities = priorities[:old_count]
        seen = int(seen_buffer.item())
        new_priorities = self._priorities(values.shape[0], seen, values.device)
        combined_values = torch.cat([old_values, values], dim=0)
        combined_priorities = torch.cat([old_priorities, new_priorities], dim=0)
        keep = min(queue.shape[0], combined_values.shape[0])
        selected = torch.topk(combined_priorities, keep, sorted=False).indices
        queue[:keep].copy_(combined_values[selected])
        priorities[:keep].copy_(combined_priorities[selected])
        count_buffer.fill_(keep)
        seen_buffer.add_(values.shape[0])

    def _cache_foreground(self, values: torch.Tensor) -> None:
        self._cache(
            values,
            self.foreground_queue,
            self.foreground_queue_priorities,
            self.foreground_queue_count,
            self.foreground_seen_count,
        )

    def _cache_background(self, values: torch.Tensor) -> None:
        self._cache(
            values,
            self.background_queue,
            self.background_queue_priorities,
            self.background_queue_count,
            self.background_seen_count,
        )

    def begin_epoch(self) -> None:
        self.foreground_queue_count.zero_()
        self.background_queue_count.zero_()
        self.foreground_seen_count.zero_()
        self.background_seen_count.zero_()
        self.foreground_queue_priorities.fill_(-1.0)
        self.background_queue_priorities.fill_(-1.0)

    def maybe_refresh_prototypes(self) -> Dict[str, float]:
        return {}

    def _sinkhorn(self, similarity: torch.Tensor) -> torch.Tensor:
        scores = similarity.detach().float() / self.sinkhorn_epsilon
        scores = scores - scores.max()
        assignments = scores.exp().t()
        assignments /= assignments.sum().clamp_min(1e-12)
        for _ in range(self.sinkhorn_iterations):
            assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
            assignments /= assignments.shape[0]
            assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
            assignments /= assignments.shape[1]
        return (assignments * assignments.shape[1]).t()

    def _farthest_initialization(
        self, values: torch.Tensor, count: int
    ) -> torch.Tensor:
        if values.shape[0] < count:
            raise RuntimeError("not enough values to initialize all prototypes")
        selected = [0]
        nearest = 1.0 - values.matmul(values[0:1].t()).squeeze(1)
        for _ in range(1, count):
            index = int(nearest.argmax().item())
            selected.append(index)
            distance = 1.0 - values.matmul(values[index:index + 1].t()).squeeze(1)
            nearest = torch.minimum(nearest, distance)
        return values[torch.tensor(selected, device=values.device)]

    def _update_bank(
        self,
        values: torch.Tensor,
        prototypes: torch.Tensor,
        dead_age: torch.Tensor,
    ) -> Dict[str, float]:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        if not bool(self.prototype_ready.item()):
            prototypes.copy_(self._farthest_initialization(values, prototypes.shape[0]))
        normalized = F.normalize(prototypes.float(), dim=-1, eps=1e-6)
        assignment = self._sinkhorn(values.matmul(normalized.t()))
        mass = assignment.sum(dim=0)
        shares = mass / mass.sum().clamp_min(1e-12)
        centroids = assignment.t().matmul(values)
        centroids = F.normalize(centroids, dim=-1, eps=1e-6)
        updated = F.normalize(
            self.prototype_momentum * normalized
            + (1.0 - self.prototype_momentum) * centroids,
            dim=-1,
            eps=1e-6,
        )
        low = shares < self.minimum_assignment_share
        dead_age.copy_(torch.where(low, dead_age + 1, torch.zeros_like(dead_age)))
        reinitialize = torch.where(dead_age >= self.dead_prototype_patience)[0]
        for prototype_index in reinitialize.tolist():
            nearest_similarity = values.matmul(updated.t()).max(dim=1).values
            candidate_index = int(nearest_similarity.argmin().item())
            updated[prototype_index] = values[candidate_index]
            dead_age[prototype_index] = 0
        prototypes.copy_(updated)
        return {
            "assignment_share_min": float(shares.min().item()),
            "effective_prototypes": _effective_count(shares),
            "pairwise_similarity_max": _pairwise_max(updated),
            "reinitialized": int(reinitialize.numel()),
        }

    @torch.no_grad()
    def finalize_epoch(self) -> Dict[str, float]:
        foreground_count = int(self.foreground_queue_count.item())
        background_count = int(self.background_queue_count.item())
        if foreground_count < self.num_fg or background_count < self.num_bg:
            return {
                "sscp_epoch_update": 0,
                "sscp_foreground_gathered": foreground_count,
                "sscp_background_gathered": background_count,
            }
        foreground = self.foreground_queue[:foreground_count]
        background = self.background_queue[:background_count]
        foreground_state = self._update_bank(
            foreground, self.foreground_prototypes, self.foreground_dead_age
        )
        background_state = self._update_bank(
            background, self.background_prototypes, self.background_dead_age
        )
        self.prototype_ready.fill_(1)
        self.prototype_update_count.add_(1)
        cross = self.normalized_foreground_prototypes().matmul(
            self.normalized_background_prototypes().t()
        )
        return {
            "sscp_epoch_update": 1,
            "sscp_foreground_gathered": foreground_count,
            "sscp_background_gathered": background_count,
            "sscp_foreground_assignment_share_min": foreground_state[
                "assignment_share_min"
            ],
            "sscp_background_assignment_share_min": background_state[
                "assignment_share_min"
            ],
            "sscp_foreground_effective_prototypes": foreground_state[
                "effective_prototypes"
            ],
            "sscp_background_effective_prototypes": background_state[
                "effective_prototypes"
            ],
            "sscp_foreground_pairwise_similarity_max": foreground_state[
                "pairwise_similarity_max"
            ],
            "sscp_background_pairwise_similarity_max": background_state[
                "pairwise_similarity_max"
            ],
            "sscp_cross_bank_similarity_max": float(cross.max().item()),
            "sscp_foreground_reinitialized": foreground_state["reinitialized"],
            "sscp_background_reinitialized": background_state["reinitialized"],
        }

    def fixed_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION.encode("utf-8"))
        digest.update(self.foreground_prototypes.detach().cpu().numpy().tobytes())
        digest.update(self.background_prototypes.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
