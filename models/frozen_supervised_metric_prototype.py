import hashlib
import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.frozen_teacher_fg_prototype import FrozenTeacherForegroundPrototype


FROZEN_SUPERVISED_METRIC_VERSION = (
    "frozen_supervised_metric_proto_v1_20260816"
)


def _logmeanexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


class FrozenSupervisedMetricPrototype(nn.Module):
    """Frozen three-class metric prototypes supervising P2P cls features."""

    def __init__(
        self,
        feat_dim: int,
        embedding_dim: int,
        prototype_counts: Tuple[int, int, int],
        temperature: float,
        positive_radius: float = 10.0,
        background_radius: float = 30.0,
        max_positive_per_image: int = 32,
        max_hard_negative_per_image: int = 16,
        max_random_negative_per_image: int = 16,
        hard_negative_weight: float = 2.0,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        if feat_dim < 1 or embedding_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if len(prototype_counts) != 3 or min(prototype_counts) < 1:
            raise ValueError("prototype_counts must contain three positive values")
        if temperature <= 0 or hard_negative_weight <= 0:
            raise ValueError("temperature and hard-negative weight must be positive")
        self.feat_dim = int(feat_dim)
        self.embedding_dim = int(embedding_dim)
        self.prototype_counts = tuple(int(value) for value in prototype_counts)
        self.temperature = float(temperature)
        self.hard_negative_weight = float(hard_negative_weight)
        offsets = [0]
        for count in self.prototype_counts:
            offsets.append(offsets[-1] + count)
        self.class_slices = tuple(
            (offsets[index], offsets[index + 1]) for index in range(3)
        )
        self.projector = nn.Linear(self.feat_dim, self.embedding_dim, bias=True)
        for parameter in self.projector.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "prototypes",
            torch.zeros(sum(self.prototype_counts), self.embedding_dim),
        )
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.bank_id = ""
        self.metadata: Dict = {}
        self._epoch_start_sha256 = ""

        # Reuse the candidate-selection contract that produced the audited archive.
        self.selector = FrozenTeacherForegroundPrototype(
            feat_dim=self.feat_dim,
            num_prototypes=1,
            negative_bank_size=1,
            negative_sample_size=1,
            temperature=self.temperature,
            positive_radius=positive_radius,
            background_radius=background_radius,
            max_positive_per_image=max_positive_per_image,
            max_hard_negative_per_image=max_hard_negative_per_image,
            max_random_negative_per_image=max_random_negative_per_image,
            sampling_seed=sampling_seed,
        )

    def _normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def embed(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        return F.normalize(self.projector(values), dim=-1, eps=1e-6)

    def class_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        similarities = embeddings.matmul(self._normalized_prototypes().t())
        logits = []
        for start, end in self.class_slices:
            logits.append(
                _logmeanexp(similarities[..., start:end] / self.temperature, dim=-1)
            )
        return torch.stack(logits, dim=-1)

    def load_bank_file(self, path: str) -> Dict:
        payload = torch.load(path, map_location="cpu")
        required = {
            "projector_weight",
            "projector_bias",
            "prototypes",
            "prototype_counts",
            "temperature",
            "bank_id",
            "metadata",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise KeyError(f"supervised prototype bank missing keys: {missing}")
        metadata = dict(payload["metadata"])
        if metadata.get("version") != FROZEN_SUPERVISED_METRIC_VERSION:
            raise ValueError("supervised prototype bank version mismatch")
        if metadata.get("gate_pass") is not True:
            raise RuntimeError("offline supervised prototype gate failed")
        if metadata.get("feature_source") != (
            "p2p_cls_features_before_final_classifier"
        ):
            raise RuntimeError("supervised prototype feature source is incompatible")
        if not metadata.get("checkpoint_sha256"):
            raise RuntimeError("supervised prototype bank lacks teacher checkpoint hash")
        if tuple(int(v) for v in payload["prototype_counts"]) != self.prototype_counts:
            raise ValueError("supervised prototype count mismatch")
        if abs(float(payload["temperature"]) - self.temperature) > 1e-8:
            raise ValueError("supervised prototype temperature mismatch")

        weight = torch.as_tensor(payload["projector_weight"]).float()
        bias = torch.as_tensor(payload["projector_bias"]).float()
        prototypes = torch.as_tensor(payload["prototypes"]).float()
        if tuple(weight.shape) != tuple(self.projector.weight.shape):
            raise ValueError("supervised projector weight shape mismatch")
        if tuple(bias.shape) != tuple(self.projector.bias.shape):
            raise ValueError("supervised projector bias shape mismatch")
        if tuple(prototypes.shape) != tuple(self.prototypes.shape):
            raise ValueError("supervised prototype tensor shape mismatch")
        if not all(torch.isfinite(value).all() for value in (weight, bias, prototypes)):
            raise ValueError("supervised prototype bank contains non-finite values")
        if (prototypes.norm(dim=-1) <= 1e-8).any():
            raise ValueError("supervised prototype bank contains a zero prototype")
        with torch.no_grad():
            self.projector.weight.copy_(weight)
            self.projector.bias.copy_(bias)
            self.prototypes.copy_(F.normalize(prototypes, dim=-1, eps=1e-6))
            self.prototype_ready.fill_(1)
        self.bank_id = str(payload["bank_id"])
        if not self.bank_id:
            raise ValueError("supervised prototype bank_id is empty")
        self.metadata = metadata
        self._epoch_start_sha256 = self.fixed_state_sha256()
        return metadata

    def fixed_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.projector.weight,
            self.projector.bias,
            self.prototypes,
        ):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(self.bank_id.encode("utf-8"))
        return digest.hexdigest()

    def begin_epoch(self) -> None:
        if not bool(self.prototype_ready.item()):
            raise RuntimeError("supervised prototype bank has not been loaded")
        self._epoch_start_sha256 = self.fixed_state_sha256()

    def finalize_epoch(self) -> Dict[str, float]:
        unchanged = self.fixed_state_sha256() == self._epoch_start_sha256
        if not unchanged:
            raise RuntimeError("frozen supervised prototype state changed")
        return {
            "supervised_proto_fixed_state_unchanged": 1.0,
            "prototype_ready": 1.0,
            "prototype_epoch_refresh_events": 0.0,
        }

    def maybe_refresh_prototypes(self, force: bool = False) -> Dict[str, float]:
        del force
        return {
            "prototype_refresh_event": 0.0,
            "prototype_refresh_count": 0.0,
            "supervised_proto_fixed_state_unchanged": 1.0,
        }

    def forward(
        self,
        features: torch.Tensor,
        raw_logits: torch.Tensor,
        apply_fusion: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if apply_fusion:
            raise RuntimeError(
                "inference fusion is disabled for frozen supervised prototypes"
            )
        embeddings = self.embed(features)
        logits3 = self.class_logits_from_embeddings(embeddings)
        background_logit = torch.logsumexp(logits3[..., 1:], dim=-1) - math.log(2.0)
        logits2 = torch.stack([logits3[..., 0], background_logit], dim=-1)
        similarities = embeddings.matmul(self._normalized_prototypes().t())
        fg_start, fg_end = self.class_slices[0]
        bg_start, _ = self.class_slices[1]
        fg_distance = 1.0 - similarities[..., fg_start:fg_end].max(dim=-1).values
        bg_distance = 1.0 - similarities[..., bg_start:].max(dim=-1).values
        return {
            "embeddings": embeddings,
            "prototype_logits": logits2,
            "prototype_distances": torch.stack([fg_distance, bg_distance], dim=-1),
            "fused_logits": raw_logits,
        }

    @staticmethod
    def _mean_or_zero(values: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        return values.mean() if values.numel() else zero

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not bool(self.prototype_ready.item()):
            raise RuntimeError("supervised prototype bank has not been loaded")
        selection = self.selector.select_candidates(
            points, raw_logits, targets, indices, anchor_points=anchor_points
        )
        logits3 = self.class_logits_from_embeddings(embeddings)
        zero = embeddings.sum() * 0.0
        positive_logits = logits3[selection.positive_mask]
        hard_logits = logits3[selection.hard_negative_mask]
        random_logits = logits3[selection.random_negative_mask]

        def class_loss(values: torch.Tensor, class_id: int) -> torch.Tensor:
            if not values.numel():
                return zero
            labels = torch.full(
                (values.shape[0],), class_id, dtype=torch.long, device=values.device
            )
            return F.cross_entropy(values, labels)

        positive_loss = class_loss(positive_logits, 0)
        hard_loss = class_loss(hard_logits, 1)
        random_loss = class_loss(random_logits, 2)
        present_weight = 0.0
        weighted = zero
        for values, value_loss, weight in (
            (positive_logits, positive_loss, 1.0),
            (hard_logits, hard_loss, self.hard_negative_weight),
            (random_logits, random_loss, 1.0),
        ):
            if values.numel():
                weighted = weighted + weight * value_loss
                present_weight += weight
        loss = weighted / max(present_weight, 1.0)

        with torch.no_grad():
            positive_margin = (
                positive_logits[:, 0] - torch.logsumexp(positive_logits[:, 1:], dim=1)
                if positive_logits.numel()
                else embeddings.new_zeros((0,))
            )
            hard_margin = (
                hard_logits[:, 0] - torch.logsumexp(hard_logits[:, 1:], dim=1)
                if hard_logits.numel()
                else embeddings.new_zeros((0,))
            )
            random_margin = (
                random_logits[:, 0] - torch.logsumexp(random_logits[:, 1:], dim=1)
                if random_logits.numel()
                else embeddings.new_zeros((0,))
            )
            diagnostics = {
                "supervised_proto_loss_raw": float(loss.detach().item()),
                "supervised_proto_positive": float(selection.positive_mask.sum().item()),
                "supervised_proto_rejected_positive": float(
                    selection.rejected_positive_mask.sum().item()
                ),
                "supervised_proto_hard_negative": float(
                    selection.hard_negative_mask.sum().item()
                ),
                "supervised_proto_random_negative": float(
                    selection.random_negative_mask.sum().item()
                ),
                "supervised_proto_ignored_near": float(
                    selection.ignored_near_mask.sum().item()
                ),
                "supervised_proto_positive_loss": float(positive_loss.detach().item()),
                "supervised_proto_hard_loss": float(hard_loss.detach().item()),
                "supervised_proto_random_loss": float(random_loss.detach().item()),
                "supervised_proto_positive_margin": float(
                    self._mean_or_zero(positive_margin, zero).detach().item()
                ),
                "supervised_proto_hard_margin": float(
                    self._mean_or_zero(hard_margin, zero).detach().item()
                ),
                "supervised_proto_random_margin": float(
                    self._mean_or_zero(random_margin, zero).detach().item()
                ),
                "prototype_ready": 1.0,
            }
        return loss, diagnostics
