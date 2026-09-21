from __future__ import annotations

import copy
import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


GT_FOREGROUND_PROXY_VERSION = "gt_foreground_proxy_v1_20260829"


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _effective_count(shares: torch.Tensor) -> float:
    positive = shares[shares > 0]
    if positive.numel() == 0:
        return 0.0
    entropy = -(positive * positive.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


class GTForegroundProxy(nn.Module):
    """Training-only foreground proxies supervised by exact GT point support."""

    def __init__(
        self,
        feat_dim: int,
        embedding_dim: int = 64,
        num_prototypes: int = 4,
        warmup_epochs: int = 5,
        support_queue_size: int = 8192,
        max_support_per_image: int = 64,
        prototype_momentum: float = 0.99,
        projector_momentum: float = 0.999,
        query_radius: float = 15.0,
        background_radius: float = 30.0,
        max_hard_background_per_image: int = 16,
        background_margin: float = 0.2,
        minimum_assignment_share: float = 0.05,
        dead_prototype_patience: int = 2,
        temperature: float = 0.2,
        align_weight: float = 1.0,
        support_weight: float = 1.0,
        query_weight: float = 1.0,
        background_weight: float = 0.5,
        balance_weight: float = 0.05,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        if feat_dim < 1 or embedding_dim < 1 or num_prototypes < 1:
            raise ValueError("feature dimensions and prototype count must be positive")
        if warmup_epochs < 0 or support_queue_size < num_prototypes:
            raise ValueError("invalid warm-up or support queue size")
        if max_support_per_image < 1:
            raise ValueError("max_support_per_image must be positive")
        if not 0.0 <= prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if not 0.0 <= projector_momentum < 1.0:
            raise ValueError("projector_momentum must be in [0, 1)")
        if query_radius <= 0 or background_radius <= query_radius:
            raise ValueError("background radius must exceed the positive query radius")
        if max_hard_background_per_image < 0:
            raise ValueError("max hard-background count must be non-negative")
        if not -1.0 < background_margin < 1.0:
            raise ValueError("background cosine margin must be in (-1, 1)")
        if not 0.0 <= minimum_assignment_share < 1.0:
            raise ValueError("minimum assignment share must be in [0, 1)")
        if dead_prototype_patience < 1 or temperature <= 0:
            raise ValueError("dead patience and temperature must be positive")

        self.feat_dim = int(feat_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_prototypes = int(num_prototypes)
        self.warmup_epochs = int(warmup_epochs)
        self.support_queue_size = int(support_queue_size)
        self.max_support_per_image = int(max_support_per_image)
        self.prototype_momentum = float(prototype_momentum)
        self.projector_momentum = float(projector_momentum)
        self.query_radius = float(query_radius)
        self.background_radius = float(background_radius)
        self.max_hard_background_per_image = int(max_hard_background_per_image)
        self.background_margin = float(background_margin)
        self.minimum_assignment_share = float(minimum_assignment_share)
        self.dead_prototype_patience = int(dead_prototype_patience)
        self.temperature = float(temperature)
        self.align_weight = float(align_weight)
        self.support_weight = float(support_weight)
        self.query_weight = float(query_weight)
        self.background_weight = float(background_weight)
        self.balance_weight = float(balance_weight)
        self.sampling_seed = int(sampling_seed)

        self.projector = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.momentum_projector = copy.deepcopy(self.projector)
        for parameter in self.momentum_projector.parameters():
            parameter.requires_grad_(False)

        self.register_buffer(
            "prototypes", torch.zeros(self.num_prototypes, self.embedding_dim)
        )
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer(
            "support_queue", torch.zeros(self.support_queue_size, self.embedding_dim)
        )
        self.register_buffer(
            "support_queue_priorities",
            torch.full((self.support_queue_size,), -1.0),
        )
        self.register_buffer("support_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("support_queue_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("support_seen_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("active_epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "dead_prototype_age", torch.zeros(self.num_prototypes, dtype=torch.long)
        )
        self.register_buffer("prototype_update_count", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _embed(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(module(values.float()), dim=-1, eps=1e-6)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def forward(
        self,
        query_features: torch.Tensor,
        raw_logits: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_embeddings = self._embed(self.projector, query_features)
        if support_features is None:
            support_embeddings = query_embeddings.new_zeros(
                query_embeddings.shape[0], 0, self.embedding_dim
            )
            momentum_support_embeddings = support_embeddings.detach()
        else:
            support_embeddings = self._embed(self.projector, support_features)
            with torch.no_grad():
                momentum_support_embeddings = self._embed(
                    self.momentum_projector, support_features.detach()
                )

        if bool(self.prototype_ready.item()):
            similarities = query_embeddings.matmul(self.normalized_prototypes().t())
            foreground_logit = similarities.max(dim=-1).values / self.temperature
        else:
            similarities = query_embeddings.new_zeros(
                *query_embeddings.shape[:-1], self.num_prototypes
            )
            foreground_logit = query_embeddings.new_zeros(query_embeddings.shape[:-1])
        prototype_logits = torch.stack([foreground_logit, -foreground_logit], dim=-1)
        prototype_distances = 1.0 - similarities
        return {
            "embeddings": query_embeddings,
            "support_embeddings": support_embeddings,
            "momentum_support_embeddings": momentum_support_embeddings,
            "prototype_logits": prototype_logits,
            "prototype_distances": prototype_distances,
            "fused_logits": raw_logits,
        }

    def _support_priorities_for_indices(
        self,
        indices: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        epoch = int(self.active_epoch.item()) if epoch is None else int(epoch)
        values = indices.to(dtype=torch.long)
        seed = self.sampling_seed + 104729 * epoch
        hashed = torch.bitwise_and(
            values * 1103515245 + (seed + 1) * 12345,
            torch.tensor(0x7FFFFFFF, dtype=torch.long, device=values.device),
        )
        return hashed.to(dtype=torch.float64) / float(0x7FFFFFFF)

    def _balanced_support_for_cache(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        selected = []
        for batch_index in range(values.shape[0]):
            local = values[batch_index, valid_mask[batch_index]]
            count = int(local.shape[0])
            if count > self.max_support_per_image:
                positions = torch.linspace(
                    0,
                    count - 1,
                    self.max_support_per_image,
                    device=local.device,
                ).round().long()
                local = local[positions]
            if local.numel():
                selected.append(local)
        if not selected:
            return values.new_zeros(0, self.embedding_dim)
        return torch.cat(selected, dim=0)

    def _cache_support(self, values: torch.Tensor) -> None:
        values = values.detach()
        if values.numel() == 0:
            return
        incoming_count = int(values.shape[0])
        seen = int(self.support_seen_count.item())
        incoming_indices = torch.arange(
            seen,
            seen + incoming_count,
            dtype=torch.long,
            device=values.device,
        )
        incoming_priorities = self._support_priorities_for_indices(incoming_indices)
        current_count = int(self.support_queue_count.item())
        combined_values = torch.cat(
            [self.support_queue[:current_count], values], dim=0
        )
        combined_priorities = torch.cat(
            [
                self.support_queue_priorities[:current_count],
                incoming_priorities.to(self.support_queue_priorities.dtype),
            ],
            dim=0,
        )
        keep = min(self.support_queue_size, int(combined_values.shape[0]))
        selected = torch.topk(combined_priorities, keep, sorted=False).indices
        self.support_queue[:keep].copy_(combined_values[selected])
        self.support_queue_priorities[:keep].copy_(combined_priorities[selected])
        if keep < self.support_queue_size:
            self.support_queue_priorities[keep:].fill_(-1.0)
        self.support_queue_count.fill_(keep)
        self.support_queue_pointer.zero_()
        self.support_seen_count.add_(incoming_count)

    def _local_support(self) -> torch.Tensor:
        count = int(self.support_queue_count.item())
        if count == 0:
            return self.support_queue[:0]
        return self.support_queue[:count]

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def compute_loss_and_cache(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        momentum_support_embeddings: torch.Tensor,
        support_valid_mask: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: Optional[torch.Tensor] = None,
        source_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del anchor_points, source_features
        zero = self._zero(query_embeddings)
        valid_support = support_embeddings[support_valid_mask]
        valid_momentum_support = momentum_support_embeddings[support_valid_mask]
        cached_support = self._balanced_support_for_cache(
            momentum_support_embeddings, support_valid_mask
        )
        self._cache_support(cached_support)

        reliable_queries: List[torch.Tensor] = []
        paired_support: List[torch.Tensor] = []
        hard_backgrounds: List[torch.Tensor] = []
        reliable_count = 0
        rejected_count = 0
        ignored_near_count = 0

        for batch_index, (source_indices, target_indices) in enumerate(indices):
            source_indices = source_indices.to(points.device)
            target_indices = target_indices.to(points.device)
            gt_points = targets["gt_points"][batch_index].to(points.device)
            matched_mask = torch.zeros(
                points.shape[1], dtype=torch.bool, device=points.device
            )
            matched_mask[source_indices] = True

            if source_indices.numel():
                matched_distances = torch.norm(
                    points[batch_index, source_indices]
                    - gt_points[target_indices],
                    dim=-1,
                )
                reliable = matched_distances <= self.query_radius
                reliable_source = source_indices[reliable]
                reliable_target = target_indices[reliable]
                reliable_count += int(reliable.sum().item())
                rejected_count += int((~reliable).sum().item())
                if reliable_source.numel():
                    reliable_queries.append(
                        query_embeddings[batch_index, reliable_source]
                    )
                    paired_support.append(
                        momentum_support_embeddings[batch_index, reliable_target]
                    )

            if gt_points.numel():
                nearest = torch.cdist(
                    points[batch_index].float(), gt_points.float(), p=2
                ).min(dim=1).values
            else:
                nearest = points.new_full(
                    (points.shape[1],), float("inf"), dtype=torch.float32
                )
            ignored_near = (~matched_mask) & (nearest <= self.background_radius)
            ignored_near_count += int(ignored_near.sum().item())
            hard_eligible = (~matched_mask) & (nearest > self.background_radius)
            hard_indices = torch.where(hard_eligible)[0]
            if hard_indices.numel() and self.max_hard_background_per_image:
                cell_scores = raw_logits[batch_index].softmax(dim=-1)[:, 0]
                count = min(
                    self.max_hard_background_per_image, int(hard_indices.numel())
                )
                selected = hard_indices[
                    torch.topk(cell_scores[hard_indices], count, sorted=False).indices
                ]
                hard_backgrounds.append(query_embeddings[batch_index, selected])

        query_values = (
            torch.cat(reliable_queries, dim=0)
            if reliable_queries
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )
        paired_values = (
            torch.cat(paired_support, dim=0)
            if paired_support
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )
        hard_values = (
            torch.cat(hard_backgrounds, dim=0)
            if hard_backgrounds
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )

        alignment_loss = (
            (1.0 - (query_values * paired_values.detach()).sum(dim=-1)).mean()
            if query_values.numel()
            else zero
        )
        support_loss = query_loss = background_loss = balance_loss = zero
        support_similarity = query_similarity = background_similarity = float("nan")

        if valid_support.numel():
            support_consistency = (
                valid_support * valid_momentum_support.detach()
            ).sum(dim=-1)
            support_loss = (1.0 - support_consistency).mean()
            support_similarity = float(support_consistency.mean().detach().item())

        if bool(self.prototype_ready.item()):
            prototypes = self.normalized_prototypes().detach()
            if valid_support.numel():
                teacher_assignment = (
                    valid_momentum_support.matmul(prototypes.t()).argmax(dim=-1)
                )
                selected_support_prototypes = prototypes[teacher_assignment]
                support_selected_similarity = (
                    valid_support * selected_support_prototypes
                ).sum(dim=-1)
                support_loss = (1.0 - support_selected_similarity).mean()
                support_probabilities = F.softmax(
                    valid_support.matmul(prototypes.t()) / self.temperature,
                    dim=-1,
                ).mean(dim=0)
                balance_loss = (
                    support_probabilities
                    * (
                        support_probabilities.clamp_min(1e-12)
                        * self.num_prototypes
                    ).log()
                ).sum()
                support_similarity = float(
                    support_selected_similarity.mean().detach().item()
                )
            if query_values.numel():
                query_similarities = query_values.matmul(prototypes.t())
                support_assignment = (
                    paired_values.detach().matmul(prototypes.t()).argmax(dim=-1)
                )
                selected_prototypes = prototypes[support_assignment]
                selected_similarity = (query_values * selected_prototypes).sum(dim=-1)
                query_loss = (1.0 - selected_similarity).mean()
                query_similarity = float(selected_similarity.mean().detach().item())
            if hard_values.numel():
                hard_max = hard_values.matmul(prototypes.t()).max(dim=-1).values
                background_loss = F.relu(
                    hard_max - self.background_margin
                ).square().mean()
                background_similarity = float(hard_max.mean().detach().item())

        loss = (
            self.align_weight * alignment_loss
            + self.support_weight * support_loss
            + self.query_weight * query_loss
            + self.background_weight * background_loss
            + self.balance_weight * balance_loss
        )
        diagnostics = {
            "prototype_ready": float(self.prototype_ready.item()),
            "gtproxy_loss_raw": float(loss.detach().item()),
            "gtproxy_loss_alignment": float(alignment_loss.detach().item()),
            "gtproxy_loss_support": float(support_loss.detach().item()),
            "gtproxy_loss_query": float(query_loss.detach().item()),
            "gtproxy_loss_background": float(background_loss.detach().item()),
            "gtproxy_loss_balance": float(balance_loss.detach().item()),
            "gtproxy_support": int(valid_support.shape[0]),
            "gtproxy_support_cached": int(cached_support.shape[0]),
            "gtproxy_reliable_query": reliable_count,
            "gtproxy_rejected_query": rejected_count,
            "gtproxy_hard_background": int(hard_values.shape[0]),
            "gtproxy_ignored_near": ignored_near_count,
            "gtproxy_support_similarity": support_similarity,
            "gtproxy_query_similarity": query_similarity,
            "gtproxy_background_similarity": background_similarity,
        }
        return loss, diagnostics

    def begin_epoch(self) -> None:
        self.support_queue_count.zero_()
        self.support_queue_pointer.zero_()
        self.support_queue_priorities.fill_(-1.0)
        self.support_seen_count.zero_()

    @torch.no_grad()
    def maybe_refresh_prototypes(self, force: bool = False) -> Dict[str, float]:
        del force
        for teacher, student in zip(
            self.momentum_projector.parameters(), self.projector.parameters()
        ):
            teacher.mul_(self.projector_momentum).add_(
                student.detach(), alpha=1.0 - self.projector_momentum
            )
        return {
            "prototype_refresh_event": 0.0,
            "prototype_refresh_count": 0.0,
            "gtproxy_momentum_projector_update": 1.0,
        }

    def _gather_support(self) -> torch.Tensor:
        local = self._local_support().detach().cpu()
        if not _dist_ready():
            return local.to(self.prototypes.device)
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        values = [value for value in gathered if value is not None and value.numel()]
        if not values:
            return self.prototypes[:0]
        return torch.cat(values, dim=0).to(self.prototypes.device)

    def _spherical_kmeans(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        first = self.sampling_seed % values.shape[0]
        centers = [values[first]]
        while len(centers) < self.num_prototypes:
            current = torch.stack(centers, dim=0)
            nearest = values.matmul(current.t()).max(dim=-1).values
            centers.append(values[nearest.argmin()])
        centers = torch.stack(centers, dim=0)
        for _ in range(20):
            assignments = values.matmul(centers.t()).argmax(dim=-1)
            updated = []
            nearest = values.matmul(centers.t()).max(dim=-1).values
            for index in range(self.num_prototypes):
                members = values[assignments == index]
                if members.numel():
                    updated.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-6))
                else:
                    updated.append(values[nearest.argmin()])
            new_centers = torch.stack(updated, dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        return F.normalize(centers, dim=-1, eps=1e-6)

    @torch.no_grad()
    def finalize_epoch(self) -> Dict[str, object]:
        support = self._gather_support()
        epoch = int(self.active_epoch.item())
        initialized = 0
        updated = 0
        reinitialized = 0
        shares = self.prototypes.new_zeros(self.num_prototypes)

        if (
            not bool(self.prototype_ready.item())
            and epoch + 1 >= self.warmup_epochs
            and support.shape[0] >= self.num_prototypes
        ):
            self.prototypes.copy_(self._spherical_kmeans(support))
            self.prototype_ready.fill_(1)
            self.dead_prototype_age.zero_()
            self.prototype_update_count.add_(1)
            initialized = 1
        elif bool(self.prototype_ready.item()) and support.numel():
            normalized = F.normalize(support.float(), dim=-1, eps=1e-6)
            prototypes = self.normalized_prototypes()
            assignments = normalized.matmul(prototypes.t()).argmax(dim=-1)
            counts = torch.bincount(
                assignments, minlength=self.num_prototypes
            ).float()
            shares = counts / counts.sum().clamp_min(1.0)
            for index in range(self.num_prototypes):
                members = normalized[assignments == index]
                if members.numel() and shares[index] >= self.minimum_assignment_share:
                    center = F.normalize(members.mean(dim=0), dim=0, eps=1e-6)
                    blended = (
                        self.prototype_momentum * prototypes[index]
                        + (1.0 - self.prototype_momentum) * center
                    )
                    self.prototypes[index].copy_(
                        F.normalize(blended, dim=0, eps=1e-6)
                    )
                    self.dead_prototype_age[index].zero_()
                else:
                    self.dead_prototype_age[index].add_(1)
            for index in range(self.num_prototypes):
                if self.dead_prototype_age[index] >= self.dead_prototype_patience:
                    current = self.normalized_prototypes()
                    least_represented = normalized.matmul(current.t()).max(dim=-1).values.argmin()
                    self.prototypes[index].copy_(normalized[least_represented])
                    self.dead_prototype_age[index].zero_()
                    reinitialized += 1
            self.prototype_update_count.add_(1)
            updated = 1

        if bool(self.prototype_ready.item()) and support.numel():
            normalized = F.normalize(support.float(), dim=-1, eps=1e-6)
            assignments = normalized.matmul(self.normalized_prototypes().t()).argmax(
                dim=-1
            )
            counts = torch.bincount(
                assignments, minlength=self.num_prototypes
            ).float()
            shares = counts / counts.sum().clamp_min(1.0)

        pairwise_max = float("nan")
        if bool(self.prototype_ready.item()) and self.num_prototypes > 1:
            pairwise = self.normalized_prototypes().matmul(
                self.normalized_prototypes().t()
            )
            pairwise = pairwise.masked_fill(
                torch.eye(
                    self.num_prototypes, dtype=torch.bool, device=pairwise.device
                ),
                -1.0,
            )
            pairwise_max = float(pairwise.max().item())

        self.active_epoch.add_(1)
        return {
            "prototype_ready": float(self.prototype_ready.item()),
            "prototype_initialization_event": float(initialized),
            "prototype_epoch_refresh_event": float(updated),
            "prototype_reinitialization_count": float(reinitialized),
            "prototype_update_count": float(self.prototype_update_count.item()),
            "gtproxy_active_epoch": float(epoch),
            "gtproxy_support_gathered": float(support.shape[0]),
            "foreground_assignment_share": [float(value) for value in shares.tolist()],
            "foreground_assignment_share_min": float(shares.min().item())
            if shares.numel()
            else 0.0,
            "foreground_effective_prototypes": _effective_count(shares),
            "foreground_pairwise_similarity_max": pairwise_max,
            "foreground_dead_age": [
                int(value) for value in self.dead_prototype_age.tolist()
            ],
        }

    def fixed_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(GT_FOREGROUND_PROXY_VERSION.encode("utf-8"))
        return digest.hexdigest()
