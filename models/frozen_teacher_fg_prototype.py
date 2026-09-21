import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


FROZEN_TEACHER_PROTOTYPE_VERSION = "ft_fgproto_v1_20260816"


@dataclass
class FrozenTeacherSelection:
    positive_mask: torch.Tensor
    rejected_positive_mask: torch.Tensor
    hard_negative_mask: torch.Tensor
    random_negative_mask: torch.Tensor
    negative_mask: torch.Tensor
    ignored_near_mask: torch.Tensor
    matched_mask: torch.Tensor
    nearest_gt_distance: torch.Tensor
    matched_distance: torch.Tensor


class FrozenTeacherForegroundPrototype(nn.Module):
    """Fixed foreground prototypes with a fixed instance-negative bank."""

    def __init__(
        self,
        feat_dim: int,
        num_prototypes: int = 4,
        negative_bank_size: int = 8192,
        negative_sample_size: int = 256,
        temperature: float = 0.1,
        positive_radius: float = 10.0,
        background_radius: float = 30.0,
        max_positive_per_image: int = 64,
        max_hard_negative_per_image: int = 16,
        max_random_negative_per_image: int = 16,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        if feat_dim < 1 or num_prototypes < 1:
            raise ValueError("feat_dim and num_prototypes must be positive")
        if negative_bank_size < 1 or negative_sample_size < 1:
            raise ValueError("negative bank sizes must be positive")
        if negative_sample_size > negative_bank_size:
            raise ValueError("negative_sample_size cannot exceed negative_bank_size")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if positive_radius <= 0 or background_radius <= positive_radius:
            raise ValueError("background_radius must exceed positive_radius")
        if min(
            max_positive_per_image,
            max_hard_negative_per_image,
            max_random_negative_per_image,
        ) < 0:
            raise ValueError("candidate limits must be non-negative")

        self.feat_dim = int(feat_dim)
        self.num_prototypes = int(num_prototypes)
        self.negative_sample_size = int(negative_sample_size)
        self.temperature = float(temperature)
        self.positive_radius = float(positive_radius)
        self.background_radius = float(background_radius)
        self.max_positive_per_image = int(max_positive_per_image)
        self.max_hard_negative_per_image = int(max_hard_negative_per_image)
        self.max_random_negative_per_image = int(max_random_negative_per_image)
        self.sampling_seed = int(sampling_seed)
        self.bank_id = ""
        self._epoch_start_sha256 = ""

        self.register_buffer(
            "fg_prototypes", torch.zeros(self.num_prototypes, self.feat_dim)
        )
        self.register_buffer(
            "negative_bank", torch.zeros(int(negative_bank_size), self.feat_dim)
        )
        self.register_buffer("negative_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("sampling_step", torch.tensor(0, dtype=torch.long))

    def load_fixed_bank(
        self,
        prototypes: torch.Tensor,
        negatives: torch.Tensor,
        bank_id: str,
    ) -> None:
        prototypes = torch.as_tensor(prototypes, dtype=self.fg_prototypes.dtype)
        negatives = torch.as_tensor(negatives, dtype=self.negative_bank.dtype)
        expected = (self.num_prototypes, self.feat_dim)
        if tuple(prototypes.shape) != expected:
            raise ValueError(
                f"foreground prototype shape must be {expected}, got {tuple(prototypes.shape)}"
            )
        if negatives.ndim != 2 or negatives.shape[1] != self.feat_dim:
            raise ValueError(
                f"negative bank must have shape [N, {self.feat_dim}]"
            )
        if negatives.shape[0] < 1:
            raise ValueError("negative bank cannot be empty")
        if negatives.shape[0] > self.negative_bank.shape[0]:
            raise ValueError(
                f"negative bank has {negatives.shape[0]} rows but capacity is "
                f"{self.negative_bank.shape[0]}"
            )
        if not bank_id:
            raise ValueError("bank_id is required")
        if not torch.isfinite(prototypes).all() or not torch.isfinite(negatives).all():
            raise ValueError("fixed bank contains non-finite values")
        if (prototypes.norm(dim=1) <= 1e-8).any():
            raise ValueError("foreground prototype cannot be zero")
        if (negatives.norm(dim=1) <= 1e-8).any():
            raise ValueError("negative bank cannot contain zero rows")

        normalized_prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
        normalized_negatives = F.normalize(negatives, dim=-1, eps=1e-6)
        with torch.no_grad():
            self.fg_prototypes.copy_(normalized_prototypes.to(self.fg_prototypes))
            self.negative_bank.zero_()
            self.negative_bank[: negatives.shape[0]].copy_(
                normalized_negatives.to(self.negative_bank)
            )
            self.negative_count.fill_(int(negatives.shape[0]))
            self.prototype_ready.fill_(1)
            self.sampling_step.zero_()
        self.bank_id = str(bank_id)
        self._epoch_start_sha256 = self.fixed_state_sha256()

    def load_fixed_bank_file(self, path: str) -> Dict:
        payload = torch.load(path, map_location="cpu")
        required = {"foreground_prototypes", "negative_bank", "bank_id"}
        missing = sorted(required.difference(payload))
        if missing:
            raise KeyError(f"fixed prototype bank is missing keys: {missing}")
        metadata = dict(payload.get("metadata", {}))
        version = metadata.get("version", FROZEN_TEACHER_PROTOTYPE_VERSION)
        if version != FROZEN_TEACHER_PROTOTYPE_VERSION:
            raise ValueError(
                f"prototype bank version mismatch: {version!r} != "
                f"{FROZEN_TEACHER_PROTOTYPE_VERSION!r}"
            )
        if metadata.get("gate_pass") is not True:
            raise RuntimeError(
                "offline foreground prototype bank did not pass its calibration gate"
            )
        if metadata.get("feature_source") != "p2p_cls_features_before_final_classifier":
            raise RuntimeError("prototype bank feature source is incompatible with P2P")
        if not metadata.get("checkpoint_sha256"):
            raise RuntimeError("prototype bank is missing its teacher checkpoint hash")
        self.load_fixed_bank(
            payload["foreground_prototypes"],
            payload["negative_bank"],
            str(payload["bank_id"]),
        )
        return metadata

    def fixed_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.fg_prototypes.detach().cpu().contiguous().numpy().tobytes())
        count = int(self.negative_count.item())
        digest.update(
            self.negative_bank[:count].detach().cpu().contiguous().numpy().tobytes()
        )
        digest.update(self.bank_id.encode("utf-8"))
        return digest.hexdigest()

    def begin_epoch(self) -> None:
        if not bool(self.prototype_ready.item()):
            raise RuntimeError("fixed teacher prototype bank has not been loaded")
        self._epoch_start_sha256 = self.fixed_state_sha256()

    def finalize_epoch(self) -> Dict[str, float]:
        unchanged = self.fixed_state_sha256() == self._epoch_start_sha256
        if not unchanged:
            raise RuntimeError("fixed teacher prototype state changed during training")
        return {
            "ft_proto_fixed_state_unchanged": 1.0,
            "prototype_ready": float(self.prototype_ready.item()),
            "prototype_epoch_refresh_events": 0.0,
        }

    def maybe_refresh_prototypes(self, force: bool = False) -> Dict[str, float]:
        del force
        return {
            "prototype_refresh_event": 0,
            "prototype_refresh_count": 0,
            "ft_proto_fixed_state_unchanged": 1.0,
        }

    def _deterministic_subset(
        self, indices: torch.Tensor, count: int, batch_idx: int
    ) -> torch.Tensor:
        if count <= 0 or indices.numel() <= count:
            return indices[: max(count, 0)] if count >= 0 else indices[:0]
        values = indices.to(dtype=torch.long)
        salt = self.sampling_seed + 104729 * int(self.sampling_step.item())
        salt += 1009 * int(batch_idx)
        keys = torch.remainder(values * 1103515245 + salt + 12345, 2147483647)
        return values[torch.argsort(keys)[:count]]

    def select_candidates(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: Optional[torch.Tensor] = None,
    ) -> FrozenTeacherSelection:
        batch_size, num_candidates = points.shape[:2]
        device = points.device
        positive = torch.zeros(batch_size, num_candidates, dtype=torch.bool, device=device)
        rejected = torch.zeros_like(positive)
        hard_negative = torch.zeros_like(positive)
        random_negative = torch.zeros_like(positive)
        ignored_near = torch.zeros_like(positive)
        matched = torch.zeros_like(positive)
        nearest = torch.full(
            (batch_size, num_candidates), float("inf"), device=device, dtype=points.dtype
        )
        matched_distance = torch.full_like(nearest, float("inf"))
        foreground_score = raw_logits.detach().softmax(dim=-1)[..., 0]

        for batch_idx, (src_idx, target_idx) in enumerate(indices):
            src_idx = src_idx.to(device=device, dtype=torch.long)
            target_idx = target_idx.to(device=device, dtype=torch.long)
            matched[batch_idx, src_idx] = True
            gt_points = targets["gt_points"][batch_idx].to(
                device=device, dtype=points.dtype
            )
            if gt_points.numel() == 0:
                far_mask = ~matched[batch_idx]
            else:
                if src_idx.numel() > 0:
                    distances = torch.linalg.vector_norm(
                        points[batch_idx, src_idx].detach() - gt_points[target_idx],
                        dim=-1,
                    )
                    matched_distance[batch_idx, src_idx] = distances
                    accepted = distances <= self.positive_radius
                    positive[batch_idx, src_idx[accepted]] = True
                    rejected[batch_idx, src_idx[~accepted]] = True
                nearest_b = torch.cdist(points[batch_idx].detach(), gt_points).min(1).values
                if anchor_points is not None:
                    anchor_nearest = torch.cdist(
                        anchor_points[batch_idx].detach(), gt_points
                    ).min(1).values
                    nearest_b = torch.minimum(nearest_b, anchor_nearest)
                nearest[batch_idx] = nearest_b
                unmatched = ~matched[batch_idx]
                ignored_near[batch_idx] = unmatched & (
                    nearest_b < self.background_radius
                )
                ignored_near[batch_idx] |= rejected[batch_idx]
                far_mask = unmatched & (nearest_b >= self.background_radius)

            far_indices = torch.where(far_mask)[0]
            hard_count = min(self.max_hard_negative_per_image, far_indices.numel())
            if hard_count > 0:
                order = torch.topk(
                    foreground_score[batch_idx, far_indices], int(hard_count)
                ).indices
                hard_negative[batch_idx, far_indices[order]] = True
            remaining = far_indices[~hard_negative[batch_idx, far_indices]]
            random_count = min(
                self.max_random_negative_per_image, int(remaining.numel())
            )
            selected_random = self._deterministic_subset(
                remaining, random_count, batch_idx
            )
            random_negative[batch_idx, selected_random] = True

        negative = hard_negative | random_negative
        if (positive & negative).any() or (rejected & negative).any():
            raise RuntimeError("frozen-teacher positive and negative selections overlap")
        if not torch.equal(positive | rejected, matched):
            raise RuntimeError("every Hungarian match must be accepted or rejected")
        if positive.any() and (
            matched_distance[positive] > self.positive_radius
        ).any():
            raise RuntimeError("positive selection violates positive_radius")
        if negative.any() and (nearest[negative] < self.background_radius).any():
            raise RuntimeError("negative selection violates background_radius")
        self.sampling_step.add_(1)
        return FrozenTeacherSelection(
            positive,
            rejected,
            hard_negative,
            random_negative,
            negative,
            ignored_near,
            matched,
            nearest,
            matched_distance,
        )

    def _sample_fixed_negatives(self) -> torch.Tensor:
        count = int(self.negative_count.item())
        if count < 1:
            raise RuntimeError("fixed negative bank is empty")
        sample_count = min(self.negative_sample_size, count)
        start = (
            self.sampling_seed
            + int(self.sampling_step.item()) * max(sample_count, 1)
        ) % count
        positions = torch.arange(sample_count, device=self.negative_bank.device)
        positions = torch.remainder(positions + start, count)
        return self.negative_bank[positions]

    @staticmethod
    def _stats(diagnostics: Dict[str, float], prefix: str, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        diagnostics[f"{prefix}_count"] = float(values.numel())
        if values.numel() == 0:
            return
        diagnostics[f"{prefix}_mean"] = float(values.mean().item())
        diagnostics[f"{prefix}_p10"] = float(torch.quantile(values, 0.1).item())
        diagnostics[f"{prefix}_p50"] = float(torch.quantile(values, 0.5).item())
        diagnostics[f"{prefix}_p90"] = float(torch.quantile(values, 0.9).item())

    def compute_loss(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not bool(self.prototype_ready.item()):
            raise RuntimeError("fixed teacher prototype bank has not been loaded")
        selection = self.select_candidates(
            points, raw_logits, targets, indices, anchor_points=anchor_points
        )
        normalized = F.normalize(features, dim=-1, eps=1e-6)
        positives = []
        for batch_idx in range(features.shape[0]):
            selected = torch.where(selection.positive_mask[batch_idx])[0]
            if self.max_positive_per_image > 0 and (
                selected.numel() > self.max_positive_per_image
            ):
                distances = selection.matched_distance[batch_idx, selected]
                selected = selected[
                    torch.argsort(distances)[: self.max_positive_per_image]
                ]
            if selected.numel() > 0:
                positives.append(normalized[batch_idx, selected])
        zero = features.sum() * 0.0
        if positives:
            queries = torch.cat(positives, dim=0)
            negatives = self._sample_fixed_negatives()
            positive_logits = queries.matmul(self.fg_prototypes.t()) / self.temperature
            negative_logits = queries.matmul(negatives.t()) / self.temperature
            positive_lse = torch.logsumexp(positive_logits, dim=1)
            all_lse = torch.logsumexp(
                torch.cat([positive_logits, negative_logits], dim=1), dim=1
            )
            loss = (all_lse - positive_lse).mean()
            nearest_positive_similarity = positive_logits.max(dim=1).values
            nearest_negative_similarity = negative_logits.max(dim=1).values
            positive_assignment = positive_logits.argmax(dim=1)
            assignment_count = torch.bincount(
                positive_assignment, minlength=self.num_prototypes
            ).float()
            assignment_share = assignment_count / assignment_count.sum().clamp_min(1.0)
            nonzero_share = assignment_share[assignment_share > 0]
            effective_prototypes = torch.exp(
                -(nonzero_share * nonzero_share.log()).sum()
            )
        else:
            loss = zero
            nearest_positive_similarity = features.new_zeros((0,))
            nearest_negative_similarity = features.new_zeros((0,))
            assignment_share = features.new_zeros((self.num_prototypes,))
            effective_prototypes = features.new_tensor(0.0)

        scores = raw_logits.detach().softmax(dim=-1)[..., 0]
        online_hard = normalized[selection.hard_negative_mask]
        online_random = normalized[selection.random_negative_mask]
        online_hard_similarity = (
            online_hard.matmul(self.fg_prototypes.t()).max(dim=1).values
            if online_hard.numel()
            else features.new_zeros((0,))
        )
        online_random_similarity = (
            online_random.matmul(self.fg_prototypes.t()).max(dim=1).values
            if online_random.numel()
            else features.new_zeros((0,))
        )
        matched_count = float(selection.matched_mask.sum().item())
        positive_count = float(selection.positive_mask.sum().item())
        positive_vs_negative_accuracy = (
            float(
                (nearest_positive_similarity > nearest_negative_similarity)
                .float()
                .mean()
                .item()
            )
            if nearest_positive_similarity.numel()
            else float("nan")
        )
        diagnostics = {
            "prototype_ready": 1.0,
            "ft_proto_positive": float(selection.positive_mask.sum().item()),
            "ft_proto_positive_used": float(
                sum(item.shape[0] for item in positives)
            ),
            "ft_proto_rejected_positive": float(
                selection.rejected_positive_mask.sum().item()
            ),
            "ft_proto_hard_negative": float(
                selection.hard_negative_mask.sum().item()
            ),
            "ft_proto_random_negative": float(
                selection.random_negative_mask.sum().item()
            ),
            "ft_proto_ignored_near": float(selection.ignored_near_mask.sum().item()),
            "ft_proto_loss_raw": float(loss.detach().item()),
            "ft_proto_online_negative_used_for_bank": 0.0,
            "ft_proto_fixed_state_unchanged": 1.0,
            "ft_proto_negative_bank_count": float(self.negative_count.item()),
            "ft_proto_negative_sample_count": float(
                min(self.negative_sample_size, int(self.negative_count.item()))
            ),
            "ft_proto_positive_acceptance_rate": (
                positive_count / matched_count if matched_count else 0.0
            ),
            "ft_proto_positive_vs_fixed_negative_accuracy": (
                positive_vs_negative_accuracy
            ),
            "ft_proto_effective_foreground_prototypes": float(
                effective_prototypes.detach().item()
            ),
        }
        for prototype_idx, share in enumerate(assignment_share.detach().tolist()):
            diagnostics[f"ft_proto_assignment_share_{prototype_idx}"] = float(share)
        self._stats(
            diagnostics,
            "ft_proto_positive_similarity",
            nearest_positive_similarity * self.temperature,
        )
        self._stats(
            diagnostics,
            "ft_proto_negative_similarity",
            nearest_negative_similarity * self.temperature,
        )
        self._stats(
            diagnostics,
            "ft_proto_similarity_gap",
            (nearest_positive_similarity - nearest_negative_similarity)
            * self.temperature,
        )
        self._stats(
            diagnostics,
            "ft_proto_online_hard_similarity",
            online_hard_similarity,
        )
        self._stats(
            diagnostics,
            "ft_proto_online_random_similarity",
            online_random_similarity,
        )
        self._stats(
            diagnostics,
            "ft_proto_raw_score_positive",
            scores[selection.positive_mask],
        )
        self._stats(
            diagnostics,
            "ft_proto_raw_score_hard_negative",
            scores[selection.hard_negative_mask],
        )
        return loss, diagnostics

    def forward(
        self,
        features: torch.Tensor,
        raw_logits: torch.Tensor,
        apply_fusion: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if apply_fusion:
            raise RuntimeError("inference fusion is forbidden for frozen-teacher v1")
        normalized = F.normalize(features, dim=-1, eps=1e-6)
        if bool(self.prototype_ready.item()):
            similarities = normalized.matmul(self.fg_prototypes.t())
        else:
            similarities = normalized.new_zeros(
                normalized.shape[:-1] + (self.num_prototypes,)
            )
        return {
            "embeddings": normalized,
            "prototype_logits": similarities,
            "prototype_distances": 1.0 - similarities,
            "fused_logits": raw_logits,
        }
