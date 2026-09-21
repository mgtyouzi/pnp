from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


PROTOTYPE_IMPLEMENTATION_VERSION = "prototype_v2_4_window_refresh_20260816"


@dataclass
class CandidateSelection:
    positive_mask: torch.Tensor
    background_mask: torch.Tensor
    ignored_near_mask: torch.Tensor
    nearest_gt_distance: torch.Tensor
    matched_mask: torch.Tensor
    rejected_positive_mask: torch.Tensor
    hard_background_mask: torch.Tensor
    random_background_mask: torch.Tensor
    matched_distance: torch.Tensor


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class CandidatePrototypeBank(nn.Module):
    """Candidate-level foreground/background prototypes for single-class P2P."""

    def __init__(
        self,
        feat_dim: int,
        embedding_dim: int = 128,
        num_fg_prototypes: int = 4,
        num_bg_prototypes: int = 4,
        foreground_queue_size: int = 4096,
        background_queue_size: int = 8192,
        temperature: float = 0.1,
        fusion_alpha: float = 0.1,
        fusion_clip: float = 2.0,
        positive_radius: float = 15.0,
        initial_positive_radius: float = 10.0,
        background_radius: float = 30.0,
        max_positive_per_image: int = 64,
        max_background_per_image: int = 32,
        max_hard_background_per_image: Optional[int] = None,
        max_random_background_per_image: int = 0,
        kmeans_iterations: int = 10,
        prototype_momentum: float = 0.99,
        dead_prototype_patience: int = 3,
        input_gradient_scale: float = 1.0,
        sampling_seed: int = 0,
        loss_mode: str = "pooled",
        positive_term_weight: float = 1.0,
        hard_background_term_weight: float = 2.0,
        random_background_term_weight: float = 0.25,
        update_mode: str = "assignment_ema",
        minimum_assignment_share: float = 0.0,
        refresh_interval_steps: int = 0,
    ) -> None:
        super().__init__()
        if num_fg_prototypes < 1 or num_bg_prototypes < 1:
            raise ValueError("prototype counts must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if positive_radius <= 0 or initial_positive_radius <= 0:
            raise ValueError("positive radii must be positive")
        if initial_positive_radius > positive_radius:
            raise ValueError("initial_positive_radius cannot exceed positive_radius")
        if background_radius <= 0:
            raise ValueError("background_radius must be positive")
        if not 0.0 <= prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if dead_prototype_patience < 1:
            raise ValueError("dead_prototype_patience must be positive")
        if not 0.0 <= input_gradient_scale <= 1.0:
            raise ValueError("input_gradient_scale must be in [0, 1]")
        if foreground_queue_size < num_fg_prototypes:
            raise ValueError("foreground queue must hold every foreground prototype")
        if background_queue_size < num_bg_prototypes:
            raise ValueError("background queue must hold every background prototype")
        if max_positive_per_image < 0:
            raise ValueError("max_positive_per_image must be non-negative")
        if max_background_per_image < 0:
            raise ValueError("max_background_per_image must be non-negative")
        if max_hard_background_per_image is not None and max_hard_background_per_image < 0:
            raise ValueError("max_hard_background_per_image must be non-negative")
        if max_random_background_per_image < 0:
            raise ValueError("max_random_background_per_image must be non-negative")
        if loss_mode not in {"pooled", "source_weighted"}:
            raise ValueError("loss_mode must be pooled or source_weighted")
        if update_mode not in {"assignment_ema", "kmeans_ema"}:
            raise ValueError("update_mode must be assignment_ema or kmeans_ema")
        term_weights = (
            positive_term_weight,
            hard_background_term_weight,
            random_background_term_weight,
        )
        if any(weight < 0 for weight in term_weights):
            raise ValueError("prototype term weights must be non-negative")
        if sum(term_weights) <= 0:
            raise ValueError("at least one prototype term weight must be positive")
        if not 0.0 <= minimum_assignment_share < 1.0:
            raise ValueError("minimum_assignment_share must be in [0, 1)")
        if refresh_interval_steps < 0:
            raise ValueError("refresh_interval_steps must be non-negative")

        self.feat_dim = int(feat_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_fg_prototypes = int(num_fg_prototypes)
        self.num_bg_prototypes = int(num_bg_prototypes)
        self.temperature = float(temperature)
        self.fusion_alpha = float(fusion_alpha)
        self.fusion_clip = float(fusion_clip)
        self.positive_radius = float(positive_radius)
        self.initial_positive_radius = float(initial_positive_radius)
        self.background_radius = float(background_radius)
        self.max_positive_per_image = int(max_positive_per_image)
        legacy_background = int(max_background_per_image)
        self.max_hard_background_per_image = int(
            legacy_background
            if max_hard_background_per_image is None
            else max_hard_background_per_image
        )
        self.max_random_background_per_image = int(max_random_background_per_image)
        self.max_background_per_image = (
            self.max_hard_background_per_image + self.max_random_background_per_image
        )
        self.kmeans_iterations = int(kmeans_iterations)
        self.prototype_momentum = float(prototype_momentum)
        self.dead_prototype_patience = int(dead_prototype_patience)
        self.input_gradient_scale = float(input_gradient_scale)
        self.sampling_seed = int(sampling_seed)
        self.loss_mode = str(loss_mode)
        self.positive_term_weight = float(positive_term_weight)
        self.hard_background_term_weight = float(hard_background_term_weight)
        self.random_background_term_weight = float(random_background_term_weight)
        self.update_mode = str(update_mode)
        self.minimum_assignment_share = float(minimum_assignment_share)
        self.refresh_interval_steps = int(refresh_interval_steps)

        self.projector = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.register_buffer(
            "fg_prototypes", torch.zeros(self.num_fg_prototypes, self.embedding_dim)
        )
        self.register_buffer(
            "bg_prototypes", torch.zeros(self.num_bg_prototypes, self.embedding_dim)
        )
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer(
            "fg_queue", torch.zeros(int(foreground_queue_size), self.embedding_dim)
        )
        self.register_buffer(
            "bg_queue", torch.zeros(int(background_queue_size), self.embedding_dim)
        )
        self.register_buffer("fg_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("bg_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("fg_queue_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("bg_queue_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "fg_dead_age", torch.zeros(self.num_fg_prototypes, dtype=torch.long)
        )
        self.register_buffer(
            "bg_dead_age", torch.zeros(self.num_bg_prototypes, dtype=torch.long)
        )
        self.register_buffer("sampling_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("refresh_window_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("center_age_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prototype_refresh_count", torch.tensor(0, dtype=torch.long))
        probe_positions = torch.arange(
            8 * self.feat_dim, dtype=torch.float32
        ).reshape(8, self.feat_dim)
        probe_features = torch.sin(0.017 * probe_positions) + torch.cos(
            0.031 * probe_positions
        )
        self.register_buffer(
            "_projector_probe_features", probe_features, persistent=False
        )
        self.register_buffer(
            "_previous_projector_probe_embeddings",
            torch.zeros(8, self.embedding_dim),
            persistent=False,
        )
        self.register_buffer(
            "_projector_probe_ready",
            torch.tensor(0, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "_previous_refresh_probe_embeddings",
            torch.zeros(8, self.embedding_dim),
            persistent=False,
        )
        self.register_buffer(
            "_refresh_probe_ready",
            torch.tensor(0, dtype=torch.uint8),
            persistent=False,
        )

        self._epoch_fg: List[torch.Tensor] = []
        self._epoch_bg: List[torch.Tensor] = []
        self._refresh_fg_embeddings: List[torch.Tensor] = []
        self._refresh_bg_embeddings: List[torch.Tensor] = []
        self._refresh_fg_features: List[torch.Tensor] = []
        self._refresh_bg_features: List[torch.Tensor] = []
        self._last_refresh_fg = torch.zeros(0, self.embedding_dim)
        self._last_refresh_bg = torch.zeros(0, self.embedding_dim)
        self._epoch_refresh_diagnostics: List[Dict[str, Any]] = []
        self._initialized_this_epoch = False

    def project(self, cls_features: torch.Tensor) -> torch.Tensor:
        projected_input = cls_features
        if self.training and self.input_gradient_scale != 1.0:
            projected_input = cls_features.detach() + self.input_gradient_scale * (
                cls_features - cls_features.detach()
            )
        return F.normalize(self.projector(projected_input), dim=-1, eps=1e-6)

    def begin_epoch(self) -> None:
        self._epoch_fg = []
        self._epoch_bg = []
        self._refresh_fg_embeddings = []
        self._refresh_bg_embeddings = []
        self._refresh_fg_features = []
        self._refresh_bg_features = []
        self._epoch_refresh_diagnostics = []
        self._initialized_this_epoch = False
        self.refresh_window_steps.zero_()

    @torch.no_grad()
    def _projector_probe_diagnostics(self) -> Dict[str, float]:
        current = F.normalize(
            self.projector(self._projector_probe_features), dim=-1, eps=1e-6
        )
        initialized_now = not bool(self._projector_probe_ready.item())
        drift_cosine = 1.0
        if not initialized_now:
            drift_cosine = float(
                F.cosine_similarity(
                    self._previous_projector_probe_embeddings,
                    current,
                    dim=-1,
                )
                .mean()
                .item()
            )
        self._previous_projector_probe_embeddings.copy_(current)
        self._projector_probe_ready.fill_(True)
        return {
            "projector_probe_initialized_now": int(initialized_now),
            "projector_probe_drift_cosine": drift_cosine,
        }

    def cache_embeddings(self, foreground: torch.Tensor, background: torch.Tensor) -> None:
        normalized_fg = (
            F.normalize(foreground.detach(), dim=-1).cpu()
            if foreground.numel() > 0
            else None
        )
        normalized_bg = (
            F.normalize(background.detach(), dim=-1).cpu()
            if background.numel() > 0
            else None
        )
        if self.refresh_interval_steps > 0:
            if normalized_fg is not None:
                self._refresh_fg_embeddings.append(normalized_fg)
            if normalized_bg is not None:
                self._refresh_bg_embeddings.append(normalized_bg)
            self.refresh_window_steps.add_(1)
            self.center_age_steps.add_(1)
            return
        if normalized_fg is not None:
            self._epoch_fg.append(normalized_fg)
        if normalized_bg is not None:
            self._epoch_bg.append(normalized_bg)

    def cache_source_features(
        self, foreground: torch.Tensor, background: torch.Tensor
    ) -> None:
        """Cache detached P2P features for projection at refresh time."""
        if self.refresh_interval_steps <= 0:
            self.cache_embeddings(self.project(foreground), self.project(background))
            return
        if foreground.numel() > 0:
            self._refresh_fg_features.append(foreground.detach().cpu())
        if background.numel() > 0:
            self._refresh_bg_features.append(background.detach().cpu())
        self.refresh_window_steps.add_(1)
        self.center_age_steps.add_(1)

    def sample_positive_indices(
        self, indices: torch.Tensor, foreground_scores: torch.Tensor
    ) -> torch.Tensor:
        """Keep score-quantile coverage instead of only easy positive matches."""
        if self.max_positive_per_image <= 0 or indices.numel() <= self.max_positive_per_image:
            return indices
        sorted_indices = indices[torch.argsort(foreground_scores[indices])]
        positions = torch.linspace(
            0,
            sorted_indices.numel() - 1,
            steps=self.max_positive_per_image,
            device=indices.device,
        ).round().long()
        return sorted_indices[positions]

    def _sample_without_global_rng(
        self,
        indices: torch.Tensor,
        count: int,
        batch_index: int,
    ) -> torch.Tensor:
        """Pseudo-random subsampling with a private CPU generator.

        The prototype branch must not advance PyTorch's global RNG because that
        would change P2P dropout and make baseline/prototype comparisons impure.
        """
        if count <= 0 or indices.numel() == 0:
            return indices[:0]
        if indices.numel() <= count:
            return indices
        modulus = (1 << 63) - 1
        seed = (
            self.sampling_seed
            + 1_000_003 * int(self.sampling_step.item())
            + 9_176 * int(batch_index)
        ) % modulus
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(
            int(indices.numel()), generator=generator, device="cpu"
        )[:count].to(device=indices.device)
        return indices[order]

    def prototype_logits_from_embeddings(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = F.normalize(embeddings.reshape(-1, self.embedding_dim), dim=-1, eps=1e-6)
        fg = F.normalize(self.fg_prototypes, dim=-1, eps=1e-6)
        bg = F.normalize(self.bg_prototypes, dim=-1, eps=1e-6)

        fg_distance = (2.0 - 2.0 * flat.matmul(fg.t())).clamp_min(0.0).min(dim=1).values
        bg_distance = (2.0 - 2.0 * flat.matmul(bg.t())).clamp_min(0.0).min(dim=1).values
        distances = torch.stack([fg_distance, bg_distance], dim=-1)
        logits = -distances / self.temperature
        shape = embeddings.shape[:-1] + (2,)
        return logits.reshape(shape), distances.reshape(shape)

    def fuse_logits(
        self,
        raw_logits: torch.Tensor,
        prototype_logits: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        scale = self.fusion_alpha if alpha is None else float(alpha)
        if not bool(self.prototype_ready.item()) or scale <= 0:
            return raw_logits
        if raw_logits.shape[-1] != 2:
            raise ValueError("prototype fusion currently supports one foreground class plus background")

        proto_log_odds = prototype_logits[..., 0] - prototype_logits[..., 1]
        correction = scale * proto_log_odds.clamp(-self.fusion_clip, self.fusion_clip)
        fused = raw_logits.clone()
        fused[..., 0] = fused[..., 0] + 0.5 * correction
        fused[..., 1] = fused[..., 1] - 0.5 * correction
        return fused

    def forward(
        self,
        cls_features: torch.Tensor,
        raw_logits: torch.Tensor,
        apply_fusion: bool = False,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self.project(cls_features)
        if bool(self.prototype_ready.item()):
            prototype_logits, prototype_distances = self.prototype_logits_from_embeddings(embeddings)
        else:
            shape = embeddings.shape[:-1] + (2,)
            prototype_logits = embeddings.new_zeros(shape)
            prototype_distances = embeddings.new_zeros(shape)
        fused = self.fuse_logits(raw_logits, prototype_logits) if apply_fusion else raw_logits
        return {
            "embeddings": embeddings,
            "prototype_logits": prototype_logits,
            "prototype_distances": prototype_distances,
            "fused_logits": fused,
        }

    def select_candidates(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: Optional[torch.Tensor] = None,
    ) -> CandidateSelection:
        batch_size, num_candidates = points.shape[:2]
        device = points.device
        positive = torch.zeros(batch_size, num_candidates, dtype=torch.bool, device=device)
        background = torch.zeros_like(positive)
        ignored_near = torch.zeros_like(positive)
        matched = torch.zeros_like(positive)
        rejected_positive = torch.zeros_like(positive)
        hard_background = torch.zeros_like(positive)
        random_background = torch.zeros_like(positive)
        nearest = torch.full(
            (batch_size, num_candidates), float("inf"), dtype=points.dtype, device=device
        )
        matched_distance = torch.full_like(nearest, float("inf"))
        foreground_score = raw_logits.detach().softmax(dim=-1)[..., 0]
        active_positive_radius = (
            self.positive_radius
            if bool(self.prototype_ready.item())
            else self.initial_positive_radius
        )

        for batch_idx, (src_idx, target_idx) in enumerate(indices):
            src_idx = src_idx.to(device=device, dtype=torch.long)
            target_idx = target_idx.to(device=device, dtype=torch.long)
            matched[batch_idx, src_idx] = True
            gt_points = targets["gt_points"][batch_idx].to(device=device, dtype=points.dtype)
            if gt_points.numel() > 0:
                if src_idx.numel() > 0:
                    assigned_gt = gt_points[target_idx]
                    assigned_distance = torch.linalg.vector_norm(
                        points[batch_idx, src_idx].detach() - assigned_gt,
                        dim=-1,
                    )
                    matched_distance[batch_idx, src_idx] = assigned_distance
                    accepted = assigned_distance <= active_positive_radius
                    positive[batch_idx, src_idx[accepted]] = True
                    rejected_positive[batch_idx, src_idx[~accepted]] = True
                nearest_b = torch.cdist(points[batch_idx].detach(), gt_points).min(dim=1).values
                if anchor_points is not None:
                    anchor_distance = torch.cdist(
                        anchor_points[batch_idx].detach(), gt_points
                    ).min(dim=1).values
                    nearest_b = torch.minimum(nearest_b, anchor_distance)
                nearest[batch_idx] = nearest_b
                unmatched = ~matched[batch_idx]
                near_mask = unmatched & (nearest_b < self.background_radius)
                far_mask = unmatched & (nearest_b >= self.background_radius)
            else:
                near_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
                far_mask = ~matched[batch_idx]
            ignored_near[batch_idx] = near_mask | rejected_positive[batch_idx]

            candidates = torch.where(far_mask)[0]
            if candidates.numel() == 0:
                continue
            hard_count = min(
                self.max_hard_background_per_image,
                int(candidates.numel()),
            )
            if hard_count > 0:
                order = torch.topk(
                    foreground_score[batch_idx, candidates], hard_count
                ).indices
                selected_hard = candidates[order]
                hard_background[batch_idx, selected_hard] = True

            remaining = candidates[~hard_background[batch_idx, candidates]]
            random_count = min(
                self.max_random_background_per_image,
                int(remaining.numel()),
            )
            if random_count > 0:
                selected_random = self._sample_without_global_rng(
                    remaining, random_count, batch_idx
                )
                random_background[batch_idx, selected_random] = True

            background[batch_idx] = (
                hard_background[batch_idx] | random_background[batch_idx]
            )

        if (positive & background).any():
            raise RuntimeError("prototype positive/background selections overlap")
        if (rejected_positive & background).any():
            raise RuntimeError("rejected prototype positives were relabelled as background")
        if not torch.equal(positive | rejected_positive, matched):
            raise RuntimeError("every Hungarian match must be accepted or explicitly rejected")
        if positive.any() and (
            matched_distance[positive] > active_positive_radius
        ).any():
            raise RuntimeError("prototype positive selection violates positive_radius")
        if rejected_positive.any() and (
            matched_distance[rejected_positive] <= active_positive_radius
        ).any():
            raise RuntimeError("prototype rejected-positive selection violates positive_radius")
        if background.any() and (nearest[background] < self.background_radius).any():
            raise RuntimeError("prototype background selection violates background_radius")

        self.sampling_step.add_(1)

        return CandidateSelection(
            positive,
            background,
            ignored_near,
            nearest,
            matched,
            rejected_positive,
            hard_background,
            random_background,
            matched_distance,
        )

    @staticmethod
    def _add_tensor_stats(
        diagnostics: Dict[str, float], prefix: str, values: torch.Tensor
    ) -> None:
        values = values.detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        diagnostics[f"{prefix}_count"] = float(values.numel())
        if values.numel() == 0:
            return
        diagnostics[f"{prefix}_mean"] = float(values.mean().item())
        diagnostics[f"{prefix}_min"] = float(values.min().item())
        diagnostics[f"{prefix}_max"] = float(values.max().item())
        for name, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            diagnostics[f"{prefix}_{name}"] = float(
                torch.quantile(values, quantile).item()
            )

    @staticmethod
    def _nearest_center_counts(
        embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> List[int]:
        if embeddings.numel() == 0:
            return [0] * int(prototypes.shape[0])
        normalized_embeddings = F.normalize(embeddings.detach(), dim=-1, eps=1e-6)
        normalized_prototypes = F.normalize(prototypes.detach(), dim=-1, eps=1e-6)
        assignment = normalized_embeddings.matmul(normalized_prototypes.t()).argmax(dim=1)
        return torch.bincount(
            assignment, minlength=int(prototypes.shape[0])
        ).cpu().tolist()

    def compute_loss_and_cache(
        self,
        embeddings: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: Optional[torch.Tensor] = None,
        source_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        selection = self.select_candidates(
            points, raw_logits, targets, indices, anchor_points=anchor_points
        )
        zero = embeddings.sum() * 0.0
        proto_logits = None
        loss_terms: Dict[str, torch.Tensor] = {}
        if bool(self.prototype_ready.item()):
            proto_logits, _ = self.prototype_logits_from_embeddings(embeddings)
            if selection.positive_mask.any():
                pos_targets = torch.zeros(
                    int(selection.positive_mask.sum().item()), dtype=torch.long, device=embeddings.device
                )
                loss_terms["positive"] = F.cross_entropy(
                    proto_logits[selection.positive_mask], pos_targets
                )

            if self.loss_mode == "pooled":
                if selection.background_mask.any():
                    bg_targets = torch.ones(
                        int(selection.background_mask.sum().item()),
                        dtype=torch.long,
                        device=embeddings.device,
                    )
                    loss_terms["background"] = F.cross_entropy(
                        proto_logits[selection.background_mask], bg_targets
                    )
                loss = (
                    torch.stack(list(loss_terms.values())).mean()
                    if loss_terms
                    else zero
                )
            else:
                for name, mask in (
                    ("hard_background", selection.hard_background_mask),
                    ("random_background", selection.random_background_mask),
                ):
                    if mask.any():
                        bg_targets = torch.ones(
                            int(mask.sum().item()),
                            dtype=torch.long,
                            device=embeddings.device,
                        )
                        loss_terms[name] = F.cross_entropy(
                            proto_logits[mask], bg_targets
                        )
                configured_weights = {
                    "positive": self.positive_term_weight,
                    "hard_background": self.hard_background_term_weight,
                    "random_background": self.random_background_term_weight,
                }
                active = [
                    (configured_weights[name], term)
                    for name, term in loss_terms.items()
                    if configured_weights[name] > 0
                ]
                if active:
                    weight_sum = sum(weight for weight, _ in active)
                    loss = sum(weight * term for weight, term in active) / weight_sum
                else:
                    loss = zero
        else:
            loss = zero

        foreground_support = []
        background_support = []
        foreground_source_support = []
        background_source_support = []
        scores = raw_logits.detach().softmax(dim=-1)[..., 0]
        for batch_idx in range(embeddings.shape[0]):
            pos_idx = torch.where(selection.positive_mask[batch_idx])[0]
            pos_idx = self.sample_positive_indices(pos_idx, scores[batch_idx])
            bg_idx = torch.where(selection.background_mask[batch_idx])[0]
            if pos_idx.numel() > 0:
                foreground_support.append(embeddings[batch_idx, pos_idx])
                if source_features is not None:
                    foreground_source_support.append(
                        source_features[batch_idx, pos_idx]
                    )
            if bg_idx.numel() > 0:
                background_support.append(embeddings[batch_idx, bg_idx])
                if source_features is not None:
                    background_source_support.append(
                        source_features[batch_idx, bg_idx]
                    )

        foreground_tensor = (
            torch.cat(foreground_support, dim=0)
            if foreground_support
            else embeddings.new_zeros((0, self.embedding_dim))
        )
        background_tensor = (
            torch.cat(background_support, dim=0)
            if background_support
            else embeddings.new_zeros((0, self.embedding_dim))
        )
        if source_features is not None and self.refresh_interval_steps > 0:
            foreground_source_tensor = (
                torch.cat(foreground_source_support, dim=0)
                if foreground_source_support
                else source_features.new_zeros((0, self.feat_dim))
            )
            background_source_tensor = (
                torch.cat(background_source_support, dim=0)
                if background_source_support
                else source_features.new_zeros((0, self.feat_dim))
            )
            self.cache_source_features(
                foreground_source_tensor, background_source_tensor
            )
        else:
            self.cache_embeddings(foreground_tensor, background_tensor)

        diagnostics = {
            "prototype_ready": float(self.prototype_ready.item()),
            "prototype_positive_radius": float(
                self.positive_radius
                if bool(self.prototype_ready.item())
                else self.initial_positive_radius
            ),
            "prototype_matched": float(selection.matched_mask.sum().item()),
            "prototype_positive": float(selection.positive_mask.sum().item()),
            "prototype_rejected_positive": float(
                selection.rejected_positive_mask.sum().item()
            ),
            "prototype_background": float(selection.background_mask.sum().item()),
            "prototype_hard_background": float(
                selection.hard_background_mask.sum().item()
            ),
            "prototype_random_background": float(
                selection.random_background_mask.sum().item()
            ),
            "prototype_ignored_near": float(selection.ignored_near_mask.sum().item()),
            "prototype_loss_raw": float(loss.detach().item()),
            "prototype_loss_mode_source_weighted": float(
                self.loss_mode == "source_weighted"
            ),
            "prototype_positive_term_weight": self.positive_term_weight,
            "prototype_hard_background_term_weight": self.hard_background_term_weight,
            "prototype_random_background_term_weight": self.random_background_term_weight,
        }
        for name, term in loss_terms.items():
            diagnostics[f"prototype_loss_{name}"] = float(term.detach().item())
        for name, mask in (
            ("positive", selection.positive_mask),
            ("rejected_positive", selection.rejected_positive_mask),
            ("ignored_near", selection.ignored_near_mask),
            ("background", selection.background_mask),
            ("hard_background", selection.hard_background_mask),
            ("random_background", selection.random_background_mask),
        ):
            self._add_tensor_stats(diagnostics, f"raw_score_{name}", scores[mask])

        finite_match_distance = selection.matched_distance[
            torch.isfinite(selection.matched_distance)
        ]
        if finite_match_distance.numel() > 0:
            for name, quantile in (("p50", 0.50), ("p90", 0.90)):
                diagnostics[f"prototype_match_distance_{name}"] = float(
                    torch.quantile(finite_match_distance.float(), quantile).item()
                )
            diagnostics["prototype_match_distance_max"] = float(
                finite_match_distance.max().item()
            )
            diagnostics["prototype_match_distance_gt15_rate"] = float(
                (finite_match_distance > 15.0).float().mean().item()
            )
            diagnostics["prototype_match_distance_gt30_rate"] = float(
                (finite_match_distance > 30.0).float().mean().item()
            )
        self._add_tensor_stats(
            diagnostics,
            "prototype_accepted_match_distance",
            selection.matched_distance[selection.positive_mask],
        )
        self._add_tensor_stats(
            diagnostics,
            "prototype_rejected_match_distance",
            selection.matched_distance[selection.rejected_positive_mask],
        )

        if proto_logits is not None:
            margin = proto_logits[..., 0] - proto_logits[..., 1]
            for name, mask in (
                ("positive", selection.positive_mask),
                ("rejected_positive", selection.rejected_positive_mask),
                ("background", selection.background_mask),
                ("hard_background", selection.hard_background_mask),
                ("random_background", selection.random_background_mask),
            ):
                self._add_tensor_stats(
                    diagnostics, f"prototype_{name}_margin", margin[mask]
                )
            if selection.positive_mask.any():
                diagnostics["prototype_positive_margin"] = diagnostics[
                    "prototype_positive_margin_mean"
                ]
                diagnostics["prototype_positive_correct_rate"] = float(
                    (margin[selection.positive_mask] > 0).float().mean().item()
                )
            if selection.background_mask.any():
                diagnostics["prototype_background_margin"] = diagnostics[
                    "prototype_background_margin_mean"
                ]
                diagnostics["prototype_background_correct_rate"] = float(
                    (margin[selection.background_mask] < 0).float().mean().item()
                )
            if selection.hard_background_mask.any():
                diagnostics["prototype_hard_background_correct_rate"] = float(
                    (margin[selection.hard_background_mask] < 0).float().mean().item()
                )
            if selection.random_background_mask.any():
                diagnostics["prototype_random_background_correct_rate"] = float(
                    (margin[selection.random_background_mask] < 0).float().mean().item()
                )

            positive_counts = self._nearest_center_counts(
                embeddings[selection.positive_mask], self.fg_prototypes
            )
            background_counts = self._nearest_center_counts(
                embeddings[selection.background_mask], self.bg_prototypes
            )
            hard_background_counts = self._nearest_center_counts(
                embeddings[selection.hard_background_mask], self.bg_prototypes
            )
            random_background_counts = self._nearest_center_counts(
                embeddings[selection.random_background_mask], self.bg_prototypes
            )
            for center_idx, count in enumerate(positive_counts):
                diagnostics[
                    f"prototype_positive_fg_center_{center_idx}_count"
                ] = float(count)
            for center_idx, count in enumerate(background_counts):
                diagnostics[
                    f"prototype_background_bg_center_{center_idx}_count"
                ] = float(count)
            for center_idx, count in enumerate(hard_background_counts):
                diagnostics[
                    f"prototype_hard_background_bg_center_{center_idx}_count"
                ] = float(count)
            for center_idx, count in enumerate(random_background_counts):
                diagnostics[
                    f"prototype_random_background_bg_center_{center_idx}_count"
                ] = float(count)
        return loss, diagnostics

    def _gather_variable(self, values: torch.Tensor) -> torch.Tensor:
        device = self.fg_prototypes.device
        values = values.to(device=device, dtype=self.fg_prototypes.dtype)
        if not _dist_ready():
            return values

        local_count = torch.tensor([values.shape[0]], dtype=torch.long, device=device)
        counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(counts, local_count)
        sizes = [int(item.item()) for item in counts]
        max_count = max(sizes) if sizes else 0
        if max_count == 0:
            return values.new_zeros((0, self.embedding_dim))

        padded = values.new_zeros((max_count, self.embedding_dim))
        if values.numel() > 0:
            padded[: values.shape[0]].copy_(values)
        gathered = [torch.zeros_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
        return torch.cat([item[:count] for item, count in zip(gathered, sizes)], dim=0)

    @staticmethod
    def _limit_samples(values: torch.Tensor, maximum: int) -> torch.Tensor:
        if maximum <= 0 or values.shape[0] <= maximum:
            return values
        positions = torch.linspace(
            0, values.shape[0] - 1, steps=maximum, device=values.device
        ).round().long()
        return values[positions]

    @staticmethod
    def _append_queue(
        queue: torch.Tensor,
        count: torch.Tensor,
        pointer: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if values.numel() == 0:
            return
        values = CandidatePrototypeBank._limit_samples(values, queue.shape[0]).to(queue)
        positions = (
            torch.arange(values.shape[0], device=queue.device) + int(pointer.item())
        ) % queue.shape[0]
        queue.index_copy_(0, positions, values)
        pointer.fill_((int(pointer.item()) + values.shape[0]) % queue.shape[0])
        count.fill_(min(queue.shape[0], int(count.item()) + values.shape[0]))

    def _kmeans(self, values: torch.Tensor, clusters: int) -> torch.Tensor:
        values = F.normalize(values, dim=-1, eps=1e-6)
        if values.shape[0] < clusters:
            raise ValueError("not enough queue entries to initialize all prototypes")

        mean = F.normalize(values.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
        first = (2.0 - 2.0 * values.matmul(mean.t()).squeeze(1)).argmin()
        selected = [int(first.item())]
        min_distance = torch.full((values.shape[0],), float("inf"), device=values.device)
        while len(selected) < clusters:
            center = values[selected[-1]].unsqueeze(0)
            distance = (2.0 - 2.0 * values.matmul(center.t()).squeeze(1)).clamp_min(0.0)
            min_distance = torch.minimum(min_distance, distance)
            min_distance[selected] = -1.0
            selected.append(int(min_distance.argmax().item()))
        centers = values[selected].clone()

        for _ in range(max(self.kmeans_iterations, 1)):
            distance = (2.0 - 2.0 * values.matmul(centers.t())).clamp_min(0.0)
            assignment = distance.argmin(dim=1)
            updated = centers.clone()
            for cluster_idx in range(clusters):
                members = values[assignment == cluster_idx]
                if members.numel() > 0:
                    updated[cluster_idx] = members.mean(dim=0)
            updated = F.normalize(updated, dim=-1, eps=1e-6)
            if torch.allclose(updated, centers, atol=1e-5, rtol=1e-5):
                centers = updated
                break
            centers = updated
        return centers

    @staticmethod
    def _within_class_similarity(prototypes: torch.Tensor) -> Tuple[float, float]:
        if prototypes.shape[0] < 2:
            return 0.0, 0.0
        normalized = F.normalize(prototypes, dim=-1, eps=1e-6)
        similarity = normalized.matmul(normalized.t())
        mask = ~torch.eye(
            prototypes.shape[0], dtype=torch.bool, device=prototypes.device
        )
        values = similarity[mask]
        return float(values.mean().item()), float(values.max().item())

    @staticmethod
    def _assignment_details(
        values: torch.Tensor, prototypes: torch.Tensor
    ) -> Tuple[List[int], List[float], List[float], float]:
        num_centers = int(prototypes.shape[0])
        if values.numel() == 0:
            return [0] * num_centers, [0.0] * num_centers, [0.0] * num_centers, 0.0
        values = F.normalize(values.to(prototypes), dim=-1, eps=1e-6)
        centers = F.normalize(prototypes, dim=-1, eps=1e-6)
        similarities = values.matmul(centers.t())
        assignment = similarities.argmax(dim=1)
        counts_tensor = torch.bincount(assignment, minlength=num_centers)
        counts = [int(value) for value in counts_tensor.cpu().tolist()]
        total = max(sum(counts), 1)
        shares = [count / total for count in counts]
        support_similarity = []
        for center_idx in range(num_centers):
            members = similarities[assignment == center_idx, center_idx]
            support_similarity.append(
                float(members.mean().item()) if members.numel() else 0.0
            )
        positive_shares = torch.tensor(
            [share for share in shares if share > 0], dtype=torch.float64
        )
        effective = float(
            torch.exp(-(positive_shares * positive_shares.log()).sum()).item()
        ) if positive_shares.numel() else 0.0
        return counts, shares, support_similarity, effective

    @torch.no_grad()
    def _ema_update_class(
        self,
        prototypes: torch.Tensor,
        dead_age: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[List[int], float, int, Dict[str, float], Optional[torch.Tensor]]:
        if values.numel() == 0:
            dead_age.add_(1)
            return [0] * prototypes.shape[0], 1.0, 0, {}, None

        values = F.normalize(values.to(prototypes), dim=-1, eps=1e-6)
        old = F.normalize(prototypes.clone(), dim=-1, eps=1e-6)
        assignment = values.matmul(old.t()).argmax(dim=1)
        counts: List[int] = []
        reinitialized = 0
        for cluster_idx in range(prototypes.shape[0]):
            members = values[assignment == cluster_idx]
            counts.append(int(members.shape[0]))
            if members.numel() > 0:
                target = F.normalize(members.mean(dim=0), dim=0, eps=1e-6)
                updated = self.prototype_momentum * old[cluster_idx] + (
                    1.0 - self.prototype_momentum
                ) * target
                prototypes[cluster_idx].copy_(
                    F.normalize(updated, dim=0, eps=1e-6)
                )
                dead_age[cluster_idx].zero_()
            else:
                dead_age[cluster_idx].add_(1)

        for cluster_idx in torch.where(
            dead_age >= self.dead_prototype_patience
        )[0].tolist():
            current = F.normalize(prototypes, dim=-1, eps=1e-6)
            closest_similarity = values.matmul(current.t()).max(dim=1).values
            replacement = values[closest_similarity.argmin()]
            prototypes[cluster_idx].copy_(replacement)
            dead_age[cluster_idx].zero_()
            reinitialized += 1

        drift = F.cosine_similarity(old, prototypes, dim=-1).mean()
        return counts, float(drift.item()), reinitialized, {}, None

    @staticmethod
    def _align_fresh_centers(
        old_centers: torch.Tensor, fresh_centers: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """Greedily align unordered KMeans centers to persistent center slots."""
        old = F.normalize(old_centers, dim=-1, eps=1e-6)
        fresh = F.normalize(fresh_centers, dim=-1, eps=1e-6)
        similarity = old.matmul(fresh.t())
        aligned = torch.zeros_like(fresh)
        fresh_for_old = [-1] * int(old.shape[0])
        available_old = torch.ones(old.shape[0], dtype=torch.bool, device=old.device)
        available_fresh = torch.ones(
            fresh.shape[0], dtype=torch.bool, device=fresh.device
        )
        for _ in range(int(old.shape[0])):
            allowed = available_old[:, None] & available_fresh[None, :]
            masked = similarity.masked_fill(~allowed, float("-inf"))
            flat = int(masked.reshape(-1).argmax().item())
            old_idx = flat // int(fresh.shape[0])
            fresh_idx = flat % int(fresh.shape[0])
            aligned[old_idx].copy_(fresh[fresh_idx])
            fresh_for_old[old_idx] = fresh_idx
            available_old[old_idx] = False
            available_fresh[fresh_idx] = False
        return aligned, fresh_for_old

    @torch.no_grad()
    def _kmeans_ema_update_class(
        self,
        prototypes: torch.Tensor,
        dead_age: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[List[int], float, int, Dict[str, float], Optional[torch.Tensor]]:
        """Update every center from a fresh current-epoch KMeans partition.

        Assignment EMA can permanently starve a center once features drift away
        from it. Fresh KMeans gives every persistent slot a current target; the
        alignment step preserves slot identity so EMA remains meaningful.
        """
        if values.numel() == 0:
            dead_age.add_(1)
            return [0] * int(prototypes.shape[0]), 1.0, 0, {}, None
        if values.shape[0] < prototypes.shape[0]:
            return self._ema_update_class(prototypes, dead_age, values)

        values = F.normalize(values.to(prototypes), dim=-1, eps=1e-6)
        old = F.normalize(prototypes.clone(), dim=-1, eps=1e-6)
        fresh = self._kmeans(values, int(prototypes.shape[0]))
        aligned, fresh_for_old = self._align_fresh_centers(old, fresh)

        fresh_assignment = values.matmul(fresh.t()).argmax(dim=1)
        fresh_counts = torch.bincount(
            fresh_assignment, minlength=int(prototypes.shape[0])
        )
        counts = [int(fresh_counts[index].item()) for index in fresh_for_old]
        minimum_count = max(
            1,
            int(math.ceil(self.minimum_assignment_share * values.shape[0])),
        )
        reinitialized = 0
        for center_idx, count in enumerate(counts):
            target = aligned[center_idx]
            updated = self.prototype_momentum * old[center_idx] + (
                1.0 - self.prototype_momentum
            ) * target
            prototypes[center_idx].copy_(F.normalize(updated, dim=0, eps=1e-6))
            if count < minimum_count:
                dead_age[center_idx].add_(1)
            else:
                dead_age[center_idx].zero_()
            if dead_age[center_idx] >= self.dead_prototype_patience:
                prototypes[center_idx].copy_(target)
                dead_age[center_idx].zero_()
                reinitialized += 1

        drift = F.cosine_similarity(old, prototypes, dim=-1).mean()
        update_diagnostics = {
            "fresh_old_alignment_cosine": float(
                F.cosine_similarity(old, aligned, dim=-1).mean().item()
            ),
            "updated_fresh_cosine": float(
                F.cosine_similarity(
                    F.normalize(prototypes, dim=-1, eps=1e-6),
                    aligned,
                    dim=-1,
                )
                .mean()
                .item()
            ),
        }
        return (
            counts,
            float(drift.item()),
            reinitialized,
            update_diagnostics,
            aligned.detach().clone(),
        )

    @torch.no_grad()
    def _materialize_refresh_window(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        has_embeddings = bool(
            self._refresh_fg_embeddings or self._refresh_bg_embeddings
        )
        has_features = bool(self._refresh_fg_features or self._refresh_bg_features)
        if has_embeddings and has_features:
            raise RuntimeError(
                "prototype refresh window cannot mix projected embeddings and source features"
            )

        if has_features:
            local_fg_features = (
                torch.cat(self._refresh_fg_features, dim=0)
                if self._refresh_fg_features
                else torch.zeros(0, self.feat_dim)
            )
            local_bg_features = (
                torch.cat(self._refresh_bg_features, dim=0)
                if self._refresh_bg_features
                else torch.zeros(0, self.feat_dim)
            )
            local_fg_features = self._limit_samples(
                local_fg_features, self.fg_queue.shape[0]
            ).to(self.fg_prototypes)
            local_bg_features = self._limit_samples(
                local_bg_features, self.bg_queue.shape[0]
            ).to(self.bg_prototypes)
            local_fg = self.project(local_fg_features)
            local_bg = self.project(local_bg_features)
            return local_fg, local_bg, 1

        local_fg = (
            torch.cat(self._refresh_fg_embeddings, dim=0)
            if self._refresh_fg_embeddings
            else torch.zeros(0, self.embedding_dim)
        )
        local_bg = (
            torch.cat(self._refresh_bg_embeddings, dim=0)
            if self._refresh_bg_embeddings
            else torch.zeros(0, self.embedding_dim)
        )
        local_fg = self._limit_samples(local_fg, self.fg_queue.shape[0])
        local_bg = self._limit_samples(local_bg, self.bg_queue.shape[0])
        return local_fg.to(self.fg_prototypes), local_bg.to(self.bg_prototypes), 0

    @torch.no_grad()
    def _refresh_probe_diagnostics(self) -> Dict[str, float]:
        current = F.normalize(
            self.projector(self._projector_probe_features), dim=-1, eps=1e-6
        )
        initialized_now = not bool(self._refresh_probe_ready.item())
        drift_cosine = 1.0
        if not initialized_now:
            drift_cosine = float(
                F.cosine_similarity(
                    self._previous_refresh_probe_embeddings,
                    current,
                    dim=-1,
                ).mean().item()
            )
        self._previous_refresh_probe_embeddings.copy_(current)
        self._refresh_probe_ready.fill_(True)
        return {
            "prototype_refresh_probe_initialized_now": int(initialized_now),
            "prototype_refresh_projector_drift_cosine": drift_cosine,
        }

    @torch.no_grad()
    def _refresh_separation_diagnostics(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        prefix: str,
    ) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}
        if not bool(self.prototype_ready.item()):
            return diagnostics
        fg_centers = F.normalize(self.fg_prototypes, dim=-1, eps=1e-6)
        bg_centers = F.normalize(self.bg_prototypes, dim=-1, eps=1e-6)
        for name, values, positive_class in (
            ("positive", foreground, True),
            ("background", background, False),
        ):
            if values.numel() == 0:
                continue
            values = F.normalize(values.to(fg_centers), dim=-1, eps=1e-6)
            fg_similarity = values.matmul(fg_centers.t()).max(dim=1).values
            bg_similarity = values.matmul(bg_centers.t()).max(dim=1).values
            margin = fg_similarity - bg_similarity
            correct = margin > 0 if positive_class else margin < 0
            diagnostics[f"{prefix}_{name}_correct_rate"] = float(
                correct.float().mean().item()
            )
            diagnostics[f"{prefix}_{name}_margin_mean"] = float(
                margin.mean().item()
            )
        return diagnostics

    @torch.no_grad()
    def maybe_refresh_prototypes(self, force: bool = False) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "prototype_refresh_event": 0,
            "prototype_refresh_interval_steps": int(self.refresh_interval_steps),
            "prototype_refresh_window_steps": int(self.refresh_window_steps.item()),
            "prototype_center_age_steps": int(self.center_age_steps.item()),
            "prototype_refresh_count": int(self.prototype_refresh_count.item()),
        }
        if self.refresh_interval_steps <= 0:
            return diagnostics

        window_steps = int(self.refresh_window_steps.item())
        due = window_steps >= self.refresh_interval_steps
        if force:
            due = window_steps > 0
        if not due:
            return diagnostics

        old_fg = self.fg_prototypes.detach().clone()
        old_bg = self.bg_prototypes.detach().clone()
        local_fg, local_bg, reprojected = self._materialize_refresh_window()
        gathered_fg = self._gather_variable(local_fg)
        gathered_bg = self._gather_variable(local_bg)
        self._last_refresh_fg = gathered_fg.detach().cpu()
        self._last_refresh_bg = gathered_bg.detach().cpu()
        ready_before = bool(self.prototype_ready.item())
        if ready_before:
            diagnostics.update(
                self._refresh_separation_diagnostics(
                    gathered_fg, gathered_bg, "prototype_refresh_pre"
                )
            )

        rank = dist.get_rank() if _dist_ready() else 0
        initialized_now = False
        fg_update_diagnostics: Dict[str, float] = {}
        bg_update_diagnostics: Dict[str, float] = {}
        if rank == 0:
            self._append_queue(
                self.fg_queue,
                self.fg_queue_count,
                self.fg_queue_pointer,
                gathered_fg,
            )
            self._append_queue(
                self.bg_queue,
                self.bg_queue_count,
                self.bg_queue_pointer,
                gathered_bg,
            )
            fg_count = int(self.fg_queue_count.item())
            bg_count = int(self.bg_queue_count.item())
            ready = (
                fg_count >= self.num_fg_prototypes
                and bg_count >= self.num_bg_prototypes
            )
            if ready and not ready_before:
                self.fg_prototypes.copy_(
                    self._kmeans(self.fg_queue[:fg_count], self.num_fg_prototypes)
                )
                self.bg_prototypes.copy_(
                    self._kmeans(self.bg_queue[:bg_count], self.num_bg_prototypes)
                )
                self.fg_dead_age.zero_()
                self.bg_dead_age.zero_()
                self.prototype_ready.fill_(True)
                initialized_now = True
            elif ready_before:
                update = (
                    self._kmeans_ema_update_class
                    if self.update_mode == "kmeans_ema"
                    else self._ema_update_class
                )
                _, _, _, fg_update_diagnostics, _ = update(
                    self.fg_prototypes, self.fg_dead_age, gathered_fg
                )
                _, _, _, bg_update_diagnostics, _ = update(
                    self.bg_prototypes, self.bg_dead_age, gathered_bg
                )
            self.prototype_refresh_count.add_(1)

        self._refresh_fg_embeddings = []
        self._refresh_bg_embeddings = []
        self._refresh_fg_features = []
        self._refresh_bg_features = []
        self.refresh_window_steps.zero_()
        self.center_age_steps.zero_()

        if _dist_ready():
            for tensor in (
                self.fg_queue,
                self.bg_queue,
                self.fg_queue_count,
                self.bg_queue_count,
                self.fg_queue_pointer,
                self.bg_queue_pointer,
                self.fg_prototypes,
                self.bg_prototypes,
                self.fg_dead_age,
                self.bg_dead_age,
                self.prototype_ready,
                self.prototype_refresh_count,
                self.refresh_window_steps,
                self.center_age_steps,
            ):
                dist.broadcast(tensor, src=0)

        diagnostics.update(
            {
                "prototype_refresh_event": 1,
                "prototype_initialized_now": int(initialized_now),
                "prototype_refresh_window_steps": window_steps,
                "prototype_refresh_foreground_gathered": int(gathered_fg.shape[0]),
                "prototype_refresh_background_gathered": int(gathered_bg.shape[0]),
                "prototype_refresh_reprojected_current_projector": int(reprojected),
                "prototype_center_age_steps_before_refresh": int(
                    diagnostics["prototype_center_age_steps"]
                ),
                "prototype_center_age_steps": 0,
                "prototype_refresh_count": int(self.prototype_refresh_count.item()),
                "prototype_refresh_foreground_center_drift_cosine": float(
                    F.cosine_similarity(
                        F.normalize(old_fg, dim=-1, eps=1e-6),
                        F.normalize(self.fg_prototypes, dim=-1, eps=1e-6),
                        dim=-1,
                    ).mean().item()
                ) if ready_before else 0.0,
                "prototype_refresh_background_center_drift_cosine": float(
                    F.cosine_similarity(
                        F.normalize(old_bg, dim=-1, eps=1e-6),
                        F.normalize(self.bg_prototypes, dim=-1, eps=1e-6),
                        dim=-1,
                    ).mean().item()
                ) if ready_before else 0.0,
                "prototype_refresh_foreground_fresh_old_alignment_cosine": (
                    fg_update_diagnostics.get("fresh_old_alignment_cosine", 1.0)
                ),
                "prototype_refresh_background_fresh_old_alignment_cosine": (
                    bg_update_diagnostics.get("fresh_old_alignment_cosine", 1.0)
                ),
            }
        )
        diagnostics.update(self._refresh_probe_diagnostics())
        diagnostics.update(
            self._refresh_separation_diagnostics(
                gathered_fg, gathered_bg, "prototype_refresh_post"
            )
        )
        if bool(self.prototype_ready.item()):
            cross = F.normalize(
                self.fg_prototypes, dim=-1, eps=1e-6
            ).matmul(F.normalize(self.bg_prototypes, dim=-1, eps=1e-6).t())
            diagnostics["prototype_refresh_cross_similarity_mean"] = float(
                cross.mean().item()
            )
            diagnostics["prototype_refresh_cross_similarity_max"] = float(
                cross.max().item()
            )
        if initialized_now:
            self._initialized_this_epoch = True
        self._epoch_refresh_diagnostics.append(dict(diagnostics))
        return diagnostics

    @torch.no_grad()
    def finalize_epoch(self) -> Dict[str, Any]:
        periodic_refresh = self.refresh_interval_steps > 0
        old_fg_prototypes = self.fg_prototypes.detach().clone()
        old_bg_prototypes = self.bg_prototypes.detach().clone()
        epoch_end_refresh: Dict[str, Any] = {}
        if periodic_refresh:
            epoch_end_refresh = self.maybe_refresh_prototypes(force=True)
            gathered_fg = self._last_refresh_fg.to(self.fg_prototypes)
            gathered_bg = self._last_refresh_bg.to(self.bg_prototypes)
        else:
            local_fg = (
                torch.cat(self._epoch_fg, dim=0)
                if self._epoch_fg
                else torch.zeros(0, self.embedding_dim)
            )
            local_bg = (
                torch.cat(self._epoch_bg, dim=0)
                if self._epoch_bg
                else torch.zeros(0, self.embedding_dim)
            )
            local_fg = self._limit_samples(local_fg, self.fg_queue.shape[0])
            local_bg = self._limit_samples(local_bg, self.bg_queue.shape[0])
            gathered_fg = self._gather_variable(local_fg)
            gathered_bg = self._gather_variable(local_bg)
        rank = dist.get_rank() if _dist_ready() else 0
        initialized_now = bool(self._initialized_this_epoch)
        fg_counts: List[int] = [0] * self.num_fg_prototypes
        bg_counts: List[int] = [0] * self.num_bg_prototypes
        fg_drift = 1.0
        bg_drift = 1.0
        fg_reinitialized = 0
        bg_reinitialized = 0
        fg_update_diagnostics: Dict[str, float] = {}
        bg_update_diagnostics: Dict[str, float] = {}
        fresh_fg_centers: Optional[torch.Tensor] = (
            self.fg_prototypes.detach().clone() if periodic_refresh else None
        )
        fresh_bg_centers: Optional[torch.Tensor] = (
            self.bg_prototypes.detach().clone() if periodic_refresh else None
        )
        if periodic_refresh:
            fg_update_diagnostics["fresh_old_alignment_cosine"] = float(
                epoch_end_refresh.get(
                    "prototype_refresh_foreground_fresh_old_alignment_cosine", 1.0
                )
            )
            bg_update_diagnostics["fresh_old_alignment_cosine"] = float(
                epoch_end_refresh.get(
                    "prototype_refresh_background_fresh_old_alignment_cosine", 1.0
                )
            )
            fg_update_diagnostics["updated_fresh_cosine"] = 1.0
            bg_update_diagnostics["updated_fresh_cosine"] = 1.0

        if rank == 0 and not periodic_refresh:
            self._append_queue(
                self.fg_queue,
                self.fg_queue_count,
                self.fg_queue_pointer,
                gathered_fg,
            )
            self._append_queue(
                self.bg_queue,
                self.bg_queue_count,
                self.bg_queue_pointer,
                gathered_bg,
            )
            fg_count = int(self.fg_queue_count.item())
            bg_count = int(self.bg_queue_count.item())
            ready = fg_count >= self.num_fg_prototypes and bg_count >= self.num_bg_prototypes
            if ready and not bool(self.prototype_ready.item()):
                self.fg_prototypes.copy_(
                    self._kmeans(self.fg_queue[:fg_count], self.num_fg_prototypes)
                )
                self.bg_prototypes.copy_(
                    self._kmeans(self.bg_queue[:bg_count], self.num_bg_prototypes)
                )
                self.fg_dead_age.zero_()
                self.bg_dead_age.zero_()
                self.prototype_ready.fill_(True)
                initialized_now = True
                fresh_fg_centers = self.fg_prototypes.detach().clone()
                fresh_bg_centers = self.bg_prototypes.detach().clone()
            elif bool(self.prototype_ready.item()):
                update = (
                    self._kmeans_ema_update_class
                    if self.update_mode == "kmeans_ema"
                    else self._ema_update_class
                )
                (
                    fg_counts,
                    fg_drift,
                    fg_reinitialized,
                    fg_update_diagnostics,
                    fresh_fg_centers,
                ) = update(
                    self.fg_prototypes,
                    self.fg_dead_age,
                    gathered_fg,
                )
                (
                    bg_counts,
                    bg_drift,
                    bg_reinitialized,
                    bg_update_diagnostics,
                    fresh_bg_centers,
                ) = update(
                    self.bg_prototypes,
                    self.bg_dead_age,
                    gathered_bg,
                )

        if _dist_ready() and not periodic_refresh:
            for tensor in (
                self.fg_queue,
                self.bg_queue,
                self.fg_queue_count,
                self.bg_queue_count,
                self.fg_queue_pointer,
                self.bg_queue_pointer,
                self.fg_prototypes,
                self.bg_prototypes,
                self.fg_dead_age,
                self.bg_dead_age,
                self.prototype_ready,
                self.sampling_step,
            ):
                dist.broadcast(tensor, src=0)

        self._epoch_fg = []
        self._epoch_bg = []
        fg_similarity_mean, fg_similarity_max = self._within_class_similarity(
            self.fg_prototypes
        )
        bg_similarity_mean, bg_similarity_max = self._within_class_similarity(
            self.bg_prototypes
        )
        cross_similarity = F.normalize(
            self.fg_prototypes, dim=-1, eps=1e-6
        ).matmul(F.normalize(self.bg_prototypes, dim=-1, eps=1e-6).t())
        fresh_cross_similarity = cross_similarity
        if fresh_fg_centers is not None and fresh_bg_centers is not None:
            fresh_cross_similarity = F.normalize(
                fresh_fg_centers, dim=-1, eps=1e-6
            ).matmul(F.normalize(fresh_bg_centers, dim=-1, eps=1e-6).t())
        fg_counts, fg_shares, fg_support_similarity, fg_effective = (
            self._assignment_details(gathered_fg, self.fg_prototypes)
        )
        bg_counts, bg_shares, bg_support_similarity, bg_effective = (
            self._assignment_details(gathered_bg, self.bg_prototypes)
        )
        fg_drift_per_center = F.cosine_similarity(
            F.normalize(old_fg_prototypes, dim=-1, eps=1e-6),
            F.normalize(self.fg_prototypes, dim=-1, eps=1e-6),
            dim=-1,
        )
        bg_drift_per_center = F.cosine_similarity(
            F.normalize(old_bg_prototypes, dim=-1, eps=1e-6),
            F.normalize(self.bg_prototypes, dim=-1, eps=1e-6),
            dim=-1,
        )
        if initialized_now:
            fg_drift_per_center.zero_()
            bg_drift_per_center.zero_()
        worst_flat = int(cross_similarity.reshape(-1).argmax().item())
        worst_fg_index = worst_flat // self.num_bg_prototypes
        worst_bg_index = worst_flat % self.num_bg_prototypes
        fg_matrix = F.normalize(
            self.fg_prototypes, dim=-1, eps=1e-6
        ).matmul(F.normalize(self.fg_prototypes, dim=-1, eps=1e-6).t())
        bg_matrix = F.normalize(
            self.bg_prototypes, dim=-1, eps=1e-6
        ).matmul(F.normalize(self.bg_prototypes, dim=-1, eps=1e-6).t())
        projector_probe = self._projector_probe_diagnostics()
        diagnostics = {
            "foreground_gathered": int(gathered_fg.shape[0]),
            "background_gathered": int(gathered_bg.shape[0]),
            "foreground_queue": int(self.fg_queue_count.item()),
            "background_queue": int(self.bg_queue_count.item()),
            "prototype_ready": int(self.prototype_ready.item()),
            "prototype_initialized_now": int(initialized_now),
            "prototype_update_mode_kmeans_ema": int(
                self.update_mode == "kmeans_ema"
            ),
            "prototype_periodic_refresh_enabled": int(periodic_refresh),
            "prototype_refresh_interval_steps": int(self.refresh_interval_steps),
            "prototype_refresh_count": int(self.prototype_refresh_count.item()),
            "prototype_center_age_steps": int(self.center_age_steps.item()),
            "prototype_epoch_end_refresh_event": int(
                epoch_end_refresh.get("prototype_refresh_event", 0)
            ),
            "minimum_assignment_share": self.minimum_assignment_share,
            "foreground_assignment_counts": fg_counts,
            "background_assignment_counts": bg_counts,
            "foreground_assignment_shares": fg_shares,
            "background_assignment_shares": bg_shares,
            "foreground_effective_prototypes": fg_effective,
            "background_effective_prototypes": bg_effective,
            "foreground_support_similarity": fg_support_similarity,
            "background_support_similarity": bg_support_similarity,
            "foreground_center_drift_cosine": float(
                fg_drift_per_center.mean().item()
            ),
            "background_center_drift_cosine": float(
                bg_drift_per_center.mean().item()
            ),
            "foreground_center_drift_per_center": fg_drift_per_center.cpu().tolist(),
            "background_center_drift_per_center": bg_drift_per_center.cpu().tolist(),
            "foreground_fresh_old_alignment_cosine": fg_update_diagnostics.get(
                "fresh_old_alignment_cosine", 1.0
            ),
            "background_fresh_old_alignment_cosine": bg_update_diagnostics.get(
                "fresh_old_alignment_cosine", 1.0
            ),
            "foreground_updated_fresh_cosine": fg_update_diagnostics.get(
                "updated_fresh_cosine", 1.0
            ),
            "background_updated_fresh_cosine": bg_update_diagnostics.get(
                "updated_fresh_cosine", 1.0
            ),
            "foreground_reinitialized": fg_reinitialized,
            "background_reinitialized": bg_reinitialized,
            "foreground_dead_ages": self.fg_dead_age.detach().cpu().tolist(),
            "background_dead_ages": self.bg_dead_age.detach().cpu().tolist(),
            "foreground_within_similarity_mean": fg_similarity_mean,
            "foreground_within_similarity_max": fg_similarity_max,
            "background_within_similarity_mean": bg_similarity_mean,
            "background_within_similarity_max": bg_similarity_max,
            "foreground_background_similarity_mean": float(
                cross_similarity.mean().item()
            ),
            "foreground_background_similarity_max": float(
                cross_similarity.max().item()
            ),
            "foreground_background_similarity_matrix": cross_similarity.cpu().tolist(),
            "fresh_foreground_background_similarity_mean": float(
                fresh_cross_similarity.mean().item()
            ),
            "fresh_foreground_background_similarity_max": float(
                fresh_cross_similarity.max().item()
            ),
            "fresh_foreground_background_similarity_matrix": (
                fresh_cross_similarity.cpu().tolist()
            ),
            "foreground_within_similarity_matrix": fg_matrix.cpu().tolist(),
            "background_within_similarity_matrix": bg_matrix.cpu().tolist(),
            "worst_cross_foreground_index": int(worst_fg_index),
            "worst_cross_background_index": int(worst_bg_index),
            "worst_cross_similarity": float(
                cross_similarity[worst_fg_index, worst_bg_index].item()
            ),
            "worst_cross_foreground_assignment_share": float(
                fg_shares[worst_fg_index]
            ),
            "worst_cross_background_assignment_share": float(
                bg_shares[worst_bg_index]
            ),
            "foreground_prototype_norms": self.fg_prototypes.norm(dim=1).cpu().tolist(),
            "background_prototype_norms": self.bg_prototypes.norm(dim=1).cpu().tolist(),
            "foreground_queue_pointer": int(self.fg_queue_pointer.item()),
            "background_queue_pointer": int(self.bg_queue_pointer.item()),
            "sampling_step": int(self.sampling_step.item()),
            "foreground_center_checksum": float(self.fg_prototypes.sum().item()),
            "background_center_checksum": float(self.bg_prototypes.sum().item()),
        }
        diagnostics.update(projector_probe)
        refresh_events = [
            item
            for item in self._epoch_refresh_diagnostics
            if int(item.get("prototype_refresh_event", 0)) == 1
        ]
        if periodic_refresh:
            def refresh_values(key: str) -> List[float]:
                return [
                    float(item[key])
                    for item in refresh_events
                    if key in item and math.isfinite(float(item[key]))
                ]

            pre_positive = refresh_values(
                "prototype_refresh_pre_positive_correct_rate"
            )
            post_positive = refresh_values(
                "prototype_refresh_post_positive_correct_rate"
            )
            pre_background = refresh_values(
                "prototype_refresh_pre_background_correct_rate"
            )
            post_background = refresh_values(
                "prototype_refresh_post_background_correct_rate"
            )
            projector_drift = refresh_values(
                "prototype_refresh_projector_drift_cosine"
            )
            center_ages = refresh_values(
                "prototype_center_age_steps_before_refresh"
            )
            reprojected = refresh_values(
                "prototype_refresh_reprojected_current_projector"
            )
            diagnostics.update(
                {
                    "prototype_epoch_refresh_events": len(refresh_events),
                    "prototype_epoch_refresh_pre_positive_correct_rate": (
                        sum(pre_positive) / len(pre_positive)
                        if pre_positive else None
                    ),
                    "prototype_epoch_refresh_post_positive_correct_rate": (
                        sum(post_positive) / len(post_positive)
                        if post_positive else None
                    ),
                    "prototype_epoch_refresh_pre_background_correct_rate": (
                        sum(pre_background) / len(pre_background)
                        if pre_background else None
                    ),
                    "prototype_epoch_refresh_post_background_correct_rate": (
                        sum(post_background) / len(post_background)
                        if post_background else None
                    ),
                    "prototype_epoch_refresh_positive_correct_gain": (
                        (sum(post_positive) / len(post_positive))
                        - (sum(pre_positive) / len(pre_positive))
                        if post_positive and pre_positive else None
                    ),
                    "prototype_epoch_refresh_background_correct_gain": (
                        (sum(post_background) / len(post_background))
                        - (sum(pre_background) / len(pre_background))
                        if post_background and pre_background else None
                    ),
                    "prototype_epoch_refresh_projector_drift_min": (
                        min(projector_drift) if projector_drift else None
                    ),
                    "prototype_epoch_refresh_center_age_max": (
                        max(center_ages) if center_ages else None
                    ),
                    "prototype_epoch_refresh_reprojected_rate": (
                        sum(reprojected) / len(reprojected)
                        if reprojected else None
                    ),
                }
            )
        return diagnostics
