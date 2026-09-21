import hashlib
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


CANDIDATE_CONDITIONED_PROTO_VERSION = (
    "candidate_conditioned_metric_proto_v2_20260829"
)

FEATURE_MODE_TO_SOURCE = {
    "cls_only": "p2p_cls_features_before_final_classifier",
    "cls_reg_context": "p2p_cls_reg_attention_offset_context",
    "spatial_morphology": (
        "p2p_fpn_p2_p3_local_spatial_center_ring_contrast"
    ),
}


def _logmeanexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


class CandidateConditionedPrototype(nn.Module):
    """Binary TP-vs-background-far prototype model for retained P2P candidates."""

    def __init__(
        self,
        feat_dim: int,
        embedding_dim: int,
        prototype_counts: Tuple[int, int],
        temperature: float,
    ) -> None:
        super().__init__()
        if feat_dim < 1 or embedding_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if len(prototype_counts) != 2 or min(prototype_counts) < 1:
            raise ValueError("prototype_counts must contain foreground and background")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.feat_dim = int(feat_dim)
        self.embedding_dim = int(embedding_dim)
        self.prototype_counts = tuple(int(value) for value in prototype_counts)
        self.temperature = float(temperature)
        self.projector = nn.Linear(self.feat_dim, self.embedding_dim, bias=True)
        self.prototypes = nn.Parameter(
            torch.empty(sum(self.prototype_counts), self.embedding_dim)
        )
        nn.init.orthogonal_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        nn.init.normal_(self.prototypes, std=0.02)
        self.class_slices = (
            (0, self.prototype_counts[0]),
            (self.prototype_counts[0], sum(self.prototype_counts)),
        )
        self.bank_id = ""
        self.metadata: Dict[str, object] = {}

    def embed(self, values: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(values.float(), dim=-1, eps=1e-6)
        return F.normalize(self.projector(normalized), dim=-1, eps=1e-6)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def class_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        similarities = embeddings.matmul(self.normalized_prototypes().t())
        logits = []
        for start, end in self.class_slices:
            logits.append(
                _logmeanexp(similarities[..., start:end] / self.temperature, dim=-1)
            )
        return torch.stack(logits, dim=-1)

    def class_logits(self, values: torch.Tensor) -> torch.Tensor:
        return self.class_logits_from_embeddings(self.embed(values))

    def margin(self, values: torch.Tensor) -> torch.Tensor:
        logits = self.class_logits(values)
        return logits[..., 0] - logits[..., 1]

    def fixed_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(self.bank_id.encode("utf-8"))
        return digest.hexdigest()

    def export_payload(self, metadata: Dict[str, object]) -> Dict[str, object]:
        state_hash = self.fixed_state_sha256()
        bank_id = f"candidate-conditioned-{state_hash[:16]}"
        return {
            "projector_weight": self.projector.weight.detach().cpu(),
            "projector_bias": self.projector.bias.detach().cpu(),
            "prototypes": self.normalized_prototypes().detach().cpu(),
            "prototype_counts": self.prototype_counts,
            "temperature": self.temperature,
            "bank_id": bank_id,
            "metadata": {
                **metadata,
                "version": CANDIDATE_CONDITIONED_PROTO_VERSION,
            },
        }

    def load_bank_file(self, path: str) -> Dict[str, object]:
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
            raise KeyError(f"candidate-conditioned bank missing keys: {missing}")
        metadata = dict(payload["metadata"])
        if metadata.get("version") != CANDIDATE_CONDITIONED_PROTO_VERSION:
            raise ValueError("candidate-conditioned bank version mismatch")
        if metadata.get("gate_pass") is not True:
            raise RuntimeError("candidate-conditioned offline gate did not pass")
        feature_mode = metadata.get("feature_mode")
        if feature_mode not in FEATURE_MODE_TO_SOURCE:
            raise RuntimeError("candidate-conditioned feature mode is incompatible")
        if metadata.get("feature_source") != FEATURE_MODE_TO_SOURCE[feature_mode]:
            raise RuntimeError("candidate-conditioned feature source is incompatible")
        if metadata.get("training_categories") != ["tp", "background_far_fp"]:
            raise RuntimeError("candidate-conditioned training labels are incompatible")
        if tuple(int(value) for value in payload["prototype_counts"]) != (
            self.prototype_counts
        ):
            raise ValueError("candidate-conditioned prototype count mismatch")
        if abs(float(payload["temperature"]) - self.temperature) > 1e-8:
            raise ValueError("candidate-conditioned temperature mismatch")

        weight = torch.as_tensor(payload["projector_weight"]).float()
        bias = torch.as_tensor(payload["projector_bias"]).float()
        prototypes = torch.as_tensor(payload["prototypes"]).float()
        if tuple(weight.shape) != tuple(self.projector.weight.shape):
            raise ValueError("candidate-conditioned projector shape mismatch")
        if tuple(bias.shape) != tuple(self.projector.bias.shape):
            raise ValueError("candidate-conditioned projector bias mismatch")
        if tuple(prototypes.shape) != tuple(self.prototypes.shape):
            raise ValueError("candidate-conditioned prototype shape mismatch")
        if not all(torch.isfinite(value).all() for value in (weight, bias, prototypes)):
            raise ValueError("candidate-conditioned bank has non-finite values")
        with torch.no_grad():
            self.projector.weight.copy_(weight)
            self.projector.bias.copy_(bias)
            self.prototypes.copy_(F.normalize(prototypes, dim=-1, eps=1e-6))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.bank_id = str(payload["bank_id"])
        self.metadata = metadata
        return metadata

    def forward(
        self,
        features: torch.Tensor,
        raw_logits: torch.Tensor,
        apply_fusion: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if apply_fusion:
            raise RuntimeError(
                "candidate-conditioned fusion requires source-calibrated gating"
            )
        embeddings = self.embed(features)
        prototype_logits = self.class_logits_from_embeddings(embeddings)
        return {
            "embeddings": embeddings,
            "prototype_logits": prototype_logits,
            "prototype_distances": 1.0 - torch.softmax(prototype_logits, dim=-1),
            "fused_logits": raw_logits,
        }
