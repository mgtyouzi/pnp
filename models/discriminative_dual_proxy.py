from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


DISCRIMINATIVE_DUAL_PROXY_VERSION = "discriminative_dual_proxy_v2_20260831"


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _effective_count(shares: torch.Tensor) -> float:
    positive = shares[shares > 0]
    if positive.numel() == 0:
        return 0.0
    entropy = -(positive * positive.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


class DiscriminativeDualProxy(nn.Module):
    """Training-only foreground/background proxies for P2P candidates."""

    def __init__(
        self,
        feat_dim: int,
        embedding_dim: int = 64,
        num_fg: int = 4,
        num_bg: int = 4,
        warmup_epochs: int = 5,
        foreground_queue_size: int = 8192,
        background_queue_size: int = 8192,
        max_foreground_per_image: int = 64,
        max_hard_background_per_image: int = 32,
        max_random_background_per_image: int = 32,
        prototype_momentum: float = 0.99,
        projector_momentum: float = 0.999,
        positive_radius: float = 15.0,
        background_radius: float = 30.0,
        temperature: float = 0.2,
        supcon_weight: float = 0.1,
        separation_weight: float = 1.0,
        separation_margin: float = 0.1,
        balance_weight: float = 0.1,
        minimum_assignment_share: float = 0.05,
        dead_prototype_patience: int = 2,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(feat_dim, embedding_dim, num_fg, num_bg) < 1:
            raise ValueError("feature dimensions and proxy counts must be positive")
        if foreground_queue_size < num_fg or background_queue_size < num_bg:
            raise ValueError("each queue must hold at least one sample per proxy")
        if min(
            max_foreground_per_image,
            max_hard_background_per_image,
            max_random_background_per_image,
        ) < 1:
            raise ValueError("per-image sample limits must be positive")
        if not 0.0 <= prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if not 0.0 <= projector_momentum < 1.0:
            raise ValueError("projector_momentum must be in [0, 1)")
        if positive_radius <= 0 or background_radius <= positive_radius:
            raise ValueError("background_radius must exceed positive_radius")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not -1.0 < separation_margin < 1.0:
            raise ValueError("separation_margin must be in (-1, 1)")
        if not 0.0 <= minimum_assignment_share < 1.0:
            raise ValueError("minimum_assignment_share must be in [0, 1)")
        if dead_prototype_patience < 1:
            raise ValueError("dead_prototype_patience must be positive")

        self.feat_dim = int(feat_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_fg = int(num_fg)
        self.num_bg = int(num_bg)
        self.warmup_epochs = int(warmup_epochs)
        self.foreground_queue_size = int(foreground_queue_size)
        self.background_queue_size = int(background_queue_size)
        self.max_foreground_per_image = int(max_foreground_per_image)
        self.max_hard_background_per_image = int(max_hard_background_per_image)
        self.max_random_background_per_image = int(max_random_background_per_image)
        self.max_background_per_image = (
            self.max_hard_background_per_image
            + self.max_random_background_per_image
        )
        self.prototype_momentum = float(prototype_momentum)
        self.projector_momentum = float(projector_momentum)
        self.positive_radius = float(positive_radius)
        self.background_radius = float(background_radius)
        self.temperature = float(temperature)
        self.supcon_weight = float(supcon_weight)
        self.separation_weight = float(separation_weight)
        self.separation_margin = float(separation_margin)
        self.balance_weight = float(balance_weight)
        self.minimum_assignment_share = float(minimum_assignment_share)
        self.dead_prototype_patience = int(dead_prototype_patience)
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
            "foreground_prototypes", torch.zeros(self.num_fg, self.embedding_dim)
        )
        self.register_buffer(
            "background_prototypes", torch.zeros(self.num_bg, self.embedding_dim)
        )
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer(
            "foreground_queue",
            torch.zeros(self.foreground_queue_size, self.embedding_dim),
        )
        self.register_buffer(
            "background_queue",
            torch.zeros(self.background_queue_size, self.embedding_dim),
        )
        self.register_buffer(
            "foreground_queue_priorities",
            torch.full((self.foreground_queue_size,), -1.0),
        )
        self.register_buffer(
            "background_queue_priorities",
            torch.full((self.background_queue_size,), -1.0),
        )
        self.register_buffer("foreground_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("background_queue_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("foreground_seen_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("background_seen_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("active_epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "foreground_dead_age", torch.zeros(self.num_fg, dtype=torch.long)
        )
        self.register_buffer(
            "background_dead_age", torch.zeros(self.num_bg, dtype=torch.long)
        )
        self.register_buffer("prototype_update_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("sampling_step", torch.tensor(0, dtype=torch.long))

    def load_bank_payload(self, payload: Dict) -> Dict[str, object]:
        if not isinstance(payload, dict):
            raise RuntimeError("dual-proxy bank payload must be a dictionary")
        if payload.get("implementation_version") != DISCRIMINATIVE_DUAL_PROXY_VERSION:
            raise RuntimeError("dual-proxy bank implementation version mismatch")
        gate = payload.get("gate", {})
        if not isinstance(gate, dict) or gate.get("gate_pass") is not True:
            raise RuntimeError("dual-proxy bank did not pass the representation gate")
        state = payload.get("prototype_state")
        if not isinstance(state, dict):
            raise RuntimeError("dual-proxy bank is missing prototype_state")
        self.load_state_dict(state, strict=True)
        if not bool(self.prototype_ready.item()):
            raise RuntimeError("loaded dual-proxy bank is not ready")
        return {
            "mode": "discriminative_dual_proxy",
            "implementation_version": DISCRIMINATIVE_DUAL_PROXY_VERSION,
            "checkpoint": str(payload.get("checkpoint", "")),
            "gate_pass": True,
            "inference_fusion": False,
        }

    def load_bank_file(self, path: str) -> Dict[str, object]:
        bank_path = Path(path)
        if not bank_path.is_file():
            raise FileNotFoundError(f"dual-proxy bank not found: {bank_path}")
        payload = torch.load(str(bank_path), map_location="cpu")
        metadata = self.load_bank_payload(payload)
        metadata["bank_path"] = str(bank_path)
        return metadata

    @staticmethod
    def _embed(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(module(values.float()), dim=-1, eps=1e-6)

    def normalized_foreground_prototypes(self) -> torch.Tensor:
        return F.normalize(self.foreground_prototypes, dim=-1, eps=1e-6)

    def normalized_background_prototypes(self) -> torch.Tensor:
        return F.normalize(self.background_prototypes, dim=-1, eps=1e-6)

    def forward(
        self,
        query_features: torch.Tensor,
        raw_logits: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_embeddings = self._embed(self.projector, query_features)
        with torch.no_grad():
            momentum_query_embeddings = self._embed(
                self.momentum_projector, query_features.detach()
            )
        if support_features is None:
            shape = (query_embeddings.shape[0], 0, self.embedding_dim)
            support_embeddings = query_embeddings.new_zeros(shape)
            momentum_support_embeddings = query_embeddings.new_zeros(shape)
        else:
            support_embeddings = self._embed(self.projector, support_features)
            with torch.no_grad():
                momentum_support_embeddings = self._embed(
                    self.momentum_projector, support_features.detach()
                )

        if bool(self.prototype_ready.item()):
            fg_similarity = query_embeddings.matmul(
                self.normalized_foreground_prototypes().t()
            ).max(dim=-1).values
            bg_similarity = query_embeddings.matmul(
                self.normalized_background_prototypes().t()
            ).max(dim=-1).values
            prototype_logits = torch.stack(
                [fg_similarity, bg_similarity], dim=-1
            ) / self.temperature
        else:
            prototype_logits = query_embeddings.new_zeros(
                *query_embeddings.shape[:-1], 2
            )
        return {
            "embeddings": query_embeddings,
            "momentum_embeddings": momentum_query_embeddings,
            "support_embeddings": support_embeddings,
            "momentum_support_embeddings": momentum_support_embeddings,
            "prototype_logits": prototype_logits,
            "prototype_distances": 1.0 - prototype_logits * self.temperature,
            "fused_logits": raw_logits,
        }

    def sample_candidates(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        sampled = {
            "reliable_indices": [],
            "reliable_target_indices": [],
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
            matched_mask = torch.zeros(
                points.shape[1], dtype=torch.bool, device=device
            )
            matched_mask[source_indices] = True

            if source_indices.numel():
                distances = torch.norm(
                    points[batch_index, source_indices].float()
                    - gt_points[target_indices],
                    dim=-1,
                )
                reliable_mask = distances <= self.positive_radius
                reliable = source_indices[reliable_mask]
                reliable_targets = target_indices[reliable_mask]
                rejected = source_indices[~reliable_mask]
            else:
                reliable = source_indices
                reliable_targets = target_indices
                rejected = source_indices

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
            if eligible_background.numel():
                cell_scores = raw_logits[batch_index].softmax(dim=-1)[:, 0]
                hard_count = min(
                    self.max_hard_background_per_image,
                    int(eligible_background.numel()),
                )
                hard = eligible_background[
                    torch.topk(
                        cell_scores[eligible_background], hard_count, sorted=False
                    ).indices
                ]
                hard_mask = torch.zeros(
                    points.shape[1], dtype=torch.bool, device=device
                )
                hard_mask[hard] = True
                remaining = eligible_background[~hard_mask[eligible_background]]
                random = self._sample_without_global_rng(
                    remaining,
                    min(
                        self.max_random_background_per_image,
                        int(remaining.numel()),
                    ),
                    batch_index,
                )
                selected = torch.cat([hard, random], dim=0)
            else:
                hard = eligible_background
                random = eligible_background
                selected = eligible_background

            sampled["reliable_indices"].append(reliable)
            sampled["reliable_target_indices"].append(reliable_targets)
            sampled["rejected_match_indices"].append(rejected)
            sampled["ignored_near_indices"].append(ignored_near)
            sampled["background_indices"].append(selected)
            sampled["hard_background_indices"].append(hard)
            sampled["random_background_indices"].append(random)
        self.sampling_step.add_(1)
        return sampled

    def _sample_without_global_rng(
        self, indices: torch.Tensor, count: int, batch_index: int
    ) -> torch.Tensor:
        """Sample without perturbing P2P dropout or augmentation RNG state."""
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

    def _priorities(
        self, indices: torch.Tensor, class_offset: int
    ) -> torch.Tensor:
        epoch = int(self.active_epoch.item())
        seed = self.sampling_seed + class_offset + 104729 * epoch
        hashed = torch.bitwise_and(
            indices.long() * 1103515245 + (seed + 1) * 12345,
            torch.tensor(0x7FFFFFFF, dtype=torch.long, device=indices.device),
        )
        return hashed.to(torch.float64) / float(0x7FFFFFFF)

    def _cache(
        self,
        values: torch.Tensor,
        queue: torch.Tensor,
        priorities: torch.Tensor,
        count_buffer: torch.Tensor,
        seen_buffer: torch.Tensor,
        class_offset: int,
    ) -> None:
        values = values.detach()
        if values.numel() == 0:
            return
        seen = int(seen_buffer.item())
        incoming_count = int(values.shape[0])
        incoming_indices = torch.arange(
            seen, seen + incoming_count, dtype=torch.long, device=values.device
        )
        incoming_priorities = self._priorities(incoming_indices, class_offset)
        current_count = int(count_buffer.item())
        combined_values = torch.cat([queue[:current_count], values], dim=0)
        combined_priorities = torch.cat(
            [priorities[:current_count], incoming_priorities.to(priorities.dtype)],
            dim=0,
        )
        keep = min(int(queue.shape[0]), int(combined_values.shape[0]))
        selected = torch.topk(combined_priorities, keep, sorted=False).indices
        queue[:keep].copy_(combined_values[selected])
        priorities[:keep].copy_(combined_priorities[selected])
        if keep < queue.shape[0]:
            priorities[keep:].fill_(-1.0)
        count_buffer.fill_(keep)
        seen_buffer.add_(incoming_count)

    def _cache_foreground(self, values: torch.Tensor) -> None:
        self._cache(
            values,
            self.foreground_queue,
            self.foreground_queue_priorities,
            self.foreground_queue_count,
            self.foreground_seen_count,
            class_offset=17,
        )

    def _cache_background(self, values: torch.Tensor) -> None:
        self._cache(
            values,
            self.background_queue,
            self.background_queue_priorities,
            self.background_queue_count,
            self.background_seen_count,
            class_offset=1000003,
        )

    def _local_foreground(self) -> torch.Tensor:
        return self.foreground_queue[: int(self.foreground_queue_count.item())]

    def _local_background(self) -> torch.Tensor:
        return self.background_queue[: int(self.background_queue_count.item())]

    def _balanced_support(
        self, values: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        selected = []
        for batch_index in range(values.shape[0]):
            local = values[batch_index, valid_mask[batch_index]]
            if local.shape[0] > self.max_foreground_per_image:
                positions = torch.linspace(
                    0,
                    local.shape[0] - 1,
                    self.max_foreground_per_image,
                    device=local.device,
                ).round().long()
                local = local[positions]
            if local.numel():
                selected.append(local)
        if not selected:
            return values.new_zeros(0, self.embedding_dim)
        return torch.cat(selected, dim=0)

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _supervised_contrastive(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return self._zero(embeddings)
        similarity = embeddings.matmul(embeddings.t()) / self.temperature
        diagonal = torch.eye(
            embeddings.shape[0], dtype=torch.bool, device=embeddings.device
        )
        valid = ~diagonal
        positive = labels[:, None].eq(labels[None, :]) & valid
        has_positive = positive.any(dim=1)
        if not bool(has_positive.any()):
            return self._zero(embeddings)
        masked_similarity = similarity.masked_fill(~valid, float("-inf"))
        log_probability = similarity - torch.logsumexp(
            masked_similarity, dim=1, keepdim=True
        )
        positive_log_probability = (
            log_probability.masked_fill(~positive, 0.0).sum(dim=1)
            / positive.sum(dim=1).clamp_min(1)
        )
        return -positive_log_probability[has_positive].mean()

    def _balance_loss(
        self, embeddings: torch.Tensor, proxies: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.numel() == 0:
            return self._zero(proxies)
        probabilities = F.softmax(
            embeddings.matmul(proxies.t()) / self.temperature, dim=-1
        ).mean(dim=0)
        return (
            probabilities
            * (probabilities.clamp_min(1e-12) * proxies.shape[0]).log()
        ).sum()

    def compute_metric_loss(
        self,
        foreground_embeddings: torch.Tensor,
        background_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        reference = (
            foreground_embeddings
            if foreground_embeddings.numel()
            else background_embeddings
        )
        zero = self._zero(reference)
        balanced_count = min(
            int(foreground_embeddings.shape[0]),
            int(background_embeddings.shape[0]),
        )
        if balanced_count > 0:
            foreground_positions = torch.linspace(
                0,
                foreground_embeddings.shape[0] - 1,
                balanced_count,
                device=foreground_embeddings.device,
            ).round().long()
            background_positions = torch.linspace(
                0,
                background_embeddings.shape[0] - 1,
                balanced_count,
                device=background_embeddings.device,
            ).round().long()
            foreground_embeddings = foreground_embeddings[foreground_positions]
            background_embeddings = background_embeddings[background_positions]
        if not bool(self.prototype_ready.item()):
            return zero, {
                "dualproxy_loss_proxy_ce": 0.0,
                "dualproxy_loss_supcon": 0.0,
                "dualproxy_loss_separation": 0.0,
                "dualproxy_loss_balance": 0.0,
                "dualproxy_loss_proxy_ce_weighted": 0.0,
                "dualproxy_loss_supcon_weighted": 0.0,
                "dualproxy_loss_separation_weighted": 0.0,
                "dualproxy_loss_balance_weighted": 0.0,
                "dualproxy_fg_similarity_gap": float("nan"),
                "dualproxy_bg_similarity_gap": float("nan"),
                "dualproxy_metric_foreground": balanced_count,
                "dualproxy_metric_background": balanced_count,
                "dualproxy_separation_fraction": 0.0,
            }
        if balanced_count == 0:
            return zero, {
                "dualproxy_loss_proxy_ce": 0.0,
                "dualproxy_loss_supcon": 0.0,
                "dualproxy_loss_separation": 0.0,
                "dualproxy_loss_balance": 0.0,
                "dualproxy_loss_proxy_ce_weighted": 0.0,
                "dualproxy_loss_supcon_weighted": 0.0,
                "dualproxy_loss_separation_weighted": 0.0,
                "dualproxy_loss_balance_weighted": 0.0,
                "dualproxy_fg_similarity_gap": float("nan"),
                "dualproxy_bg_similarity_gap": float("nan"),
                "dualproxy_metric_foreground": 0,
                "dualproxy_metric_background": 0,
                "dualproxy_separation_fraction": 0.0,
            }

        fg_proxies = self.normalized_foreground_prototypes().detach()
        bg_proxies = self.normalized_background_prototypes().detach()
        all_proxies = torch.cat([fg_proxies, bg_proxies], dim=0)
        embeddings = torch.cat(
            [foreground_embeddings, background_embeddings], dim=0
        )
        labels = torch.cat(
            [
                torch.zeros(
                    foreground_embeddings.shape[0],
                    dtype=torch.long,
                    device=embeddings.device,
                ),
                torch.ones(
                    background_embeddings.shape[0],
                    dtype=torch.long,
                    device=embeddings.device,
                ),
            ]
        )

        foreground_targets = foreground_embeddings.matmul(
            fg_proxies.t()
        ).argmax(dim=-1)
        background_targets = self.num_fg + background_embeddings.matmul(
            bg_proxies.t()
        ).argmax(dim=-1)
        foreground_proxy_ce = F.cross_entropy(
            foreground_embeddings.matmul(all_proxies.t()) / self.temperature,
            foreground_targets,
        )
        background_proxy_ce = F.cross_entropy(
            background_embeddings.matmul(all_proxies.t()) / self.temperature,
            background_targets,
        )
        proxy_ce = 0.5 * (foreground_proxy_ce + background_proxy_ce)
        supcon = self._supervised_contrastive(embeddings, labels)
        balance = self._balance_loss(
            foreground_embeddings, fg_proxies
        ) + self._balance_loss(background_embeddings, bg_proxies)

        fg_gap = bg_gap = float("nan")
        separation_terms = []
        if foreground_embeddings.numel():
            fg_own = foreground_embeddings.matmul(fg_proxies.t()).max(dim=-1).values
            fg_other = foreground_embeddings.matmul(bg_proxies.t()).max(dim=-1).values
            foreground_gap_tensor = fg_own - fg_other
            separation_terms.append(
                F.relu(self.separation_margin - foreground_gap_tensor).mean()
            )
            fg_gap = float(foreground_gap_tensor.mean().detach().item())
        if background_embeddings.numel():
            bg_own = background_embeddings.matmul(bg_proxies.t()).max(dim=-1).values
            bg_other = background_embeddings.matmul(fg_proxies.t()).max(dim=-1).values
            background_gap_tensor = bg_own - bg_other
            separation_terms.append(
                F.relu(self.separation_margin - background_gap_tensor).mean()
            )
            bg_gap = float(background_gap_tensor.mean().detach().item())
        separation = (
            torch.stack(separation_terms).mean() if separation_terms else zero
        )

        weighted_proxy_ce = proxy_ce
        weighted_supcon = self.supcon_weight * supcon
        weighted_separation = self.separation_weight * separation
        weighted_balance = self.balance_weight * balance
        loss = (
            weighted_proxy_ce
            + weighted_supcon
            + weighted_separation
            + weighted_balance
        )
        contribution_total = (
            weighted_proxy_ce.detach().abs()
            + weighted_supcon.detach().abs()
            + weighted_separation.detach().abs()
            + weighted_balance.detach().abs()
        ).clamp_min(1e-12)
        return loss, {
            "dualproxy_loss_proxy_ce": float(proxy_ce.detach().item()),
            "dualproxy_loss_supcon": float(supcon.detach().item()),
            "dualproxy_loss_separation": float(separation.detach().item()),
            "dualproxy_loss_balance": float(balance.detach().item()),
            "dualproxy_loss_proxy_ce_weighted": float(weighted_proxy_ce.detach().item()),
            "dualproxy_loss_supcon_weighted": float(weighted_supcon.detach().item()),
            "dualproxy_loss_separation_weighted": float(weighted_separation.detach().item()),
            "dualproxy_loss_balance_weighted": float(weighted_balance.detach().item()),
            "dualproxy_separation_fraction": float(
                weighted_separation.detach().abs().div(contribution_total).item()
            ),
            "dualproxy_fg_similarity_gap": fg_gap,
            "dualproxy_bg_similarity_gap": bg_gap,
            "dualproxy_metric_foreground": balanced_count,
            "dualproxy_metric_background": balanced_count,
        }

    def compute_loss_and_cache(
        self,
        query_embeddings: torch.Tensor,
        momentum_query_embeddings: torch.Tensor,
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
        sampled = self.sample_candidates(points, raw_logits, targets, indices)
        reliable = []
        momentum_reliable = []
        cached_foreground_per_image = []
        background = []
        momentum_background = []
        for batch_index in range(points.shape[0]):
            reliable_indices = sampled["reliable_indices"][batch_index]
            background_indices = sampled["background_indices"][batch_index]
            if reliable_indices.numel():
                reliable.append(query_embeddings[batch_index, reliable_indices])
                momentum_reliable.append(
                    momentum_query_embeddings[batch_index, reliable_indices]
                )
            local_foreground = momentum_support_embeddings[
                batch_index, support_valid_mask[batch_index]
            ]
            if reliable_indices.numel():
                local_foreground = torch.cat(
                    [local_foreground, momentum_reliable[-1]],
                    dim=0,
                )
            if local_foreground.shape[0] > self.max_foreground_per_image:
                positions = torch.linspace(
                    0,
                    local_foreground.shape[0] - 1,
                    self.max_foreground_per_image,
                    device=local_foreground.device,
                ).round().long()
                local_foreground = local_foreground[positions]
            if local_foreground.numel():
                cached_foreground_per_image.append(local_foreground)
            if background_indices.numel():
                background.append(query_embeddings[batch_index, background_indices])
                momentum_background.append(
                    momentum_query_embeddings[batch_index, background_indices]
                )
        cached_foreground = (
            torch.cat(cached_foreground_per_image, dim=0)
            if cached_foreground_per_image
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )
        self._cache_foreground(cached_foreground)
        reliable_values = (
            torch.cat(reliable, dim=0)
            if reliable
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )
        valid_support = support_embeddings[support_valid_mask]
        foreground_values = torch.cat(
            [valid_support, reliable_values], dim=0
        )
        background_values = (
            torch.cat(background, dim=0)
            if background
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )
        cached_background = (
            torch.cat(momentum_background, dim=0)
            if momentum_background
            else query_embeddings.new_zeros(0, self.embedding_dim)
        )
        self._cache_background(cached_background)

        metric_loss, diagnostics = self.compute_metric_loss(
            foreground_values, background_values
        )
        diagnostics.update(
            {
                "prototype_ready": float(self.prototype_ready.item()),
                "dualproxy_loss_raw": float(metric_loss.detach().item()),
                "dualproxy_gt_support": int(valid_support.shape[0]),
                "dualproxy_reliable_query": sum(
                    int(value.numel()) for value in sampled["reliable_indices"]
                ),
                "dualproxy_rejected_query": sum(
                    int(value.numel())
                    for value in sampled["rejected_match_indices"]
                ),
                "dualproxy_ignored_near": sum(
                    int(value.numel()) for value in sampled["ignored_near_indices"]
                ),
                "dualproxy_hard_background": sum(
                    int(value.numel())
                    for value in sampled["hard_background_indices"]
                ),
                "dualproxy_random_background": sum(
                    int(value.numel())
                    for value in sampled["random_background_indices"]
                ),
                "dualproxy_foreground_cached": int(cached_foreground.shape[0]),
                "dualproxy_background_cached": int(cached_background.shape[0]),
            }
        )
        return metric_loss, diagnostics

    def begin_epoch(self) -> None:
        self.foreground_queue_count.zero_()
        self.background_queue_count.zero_()
        self.foreground_seen_count.zero_()
        self.background_seen_count.zero_()
        self.foreground_queue_priorities.fill_(-1.0)
        self.background_queue_priorities.fill_(-1.0)

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
            "dualproxy_momentum_projector_update": 1.0,
        }

    def _gather(self, values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        local = values.detach().cpu()
        if not _dist_ready():
            return local.to(target.device)
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        valid = [value for value in gathered if value is not None and value.numel()]
        if not valid:
            return target[:0]
        return torch.cat(valid, dim=0).to(target.device)

    def _spherical_kmeans(
        self, values: torch.Tensor, count: int, class_offset: int
    ) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        first = (self.sampling_seed + class_offset) % values.shape[0]
        centers = [values[first]]
        while len(centers) < count:
            current = torch.stack(centers, dim=0)
            nearest = values.matmul(current.t()).max(dim=-1).values
            centers.append(values[nearest.argmin()])
        centers = torch.stack(centers, dim=0)
        for _ in range(20):
            assignments = values.matmul(centers.t()).argmax(dim=-1)
            updated = []
            nearest = values.matmul(centers.t()).max(dim=-1).values
            for index in range(count):
                members = values[assignments == index]
                updated.append(
                    F.normalize(members.mean(dim=0), dim=0, eps=1e-6)
                    if members.numel()
                    else values[nearest.argmin()]
                )
            new_centers = torch.stack(updated, dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        return F.normalize(centers, dim=-1, eps=1e-6)

    def _refresh_bank(
        self,
        values: torch.Tensor,
        prototypes: torch.Tensor,
        dead_age: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        normalized = F.normalize(values.float(), dim=-1, eps=1e-6)
        current = F.normalize(prototypes, dim=-1, eps=1e-6)
        assignments = normalized.matmul(current.t()).argmax(dim=-1)
        counts = torch.bincount(assignments, minlength=prototypes.shape[0]).float()
        shares = counts / counts.sum().clamp_min(1.0)
        reinitialized = 0
        for index in range(prototypes.shape[0]):
            members = normalized[assignments == index]
            if members.numel() and shares[index] >= self.minimum_assignment_share:
                center = F.normalize(members.mean(dim=0), dim=0, eps=1e-6)
                blended = self.prototype_momentum * current[index] + (
                    1.0 - self.prototype_momentum
                ) * center
                prototypes[index].copy_(F.normalize(blended, dim=0, eps=1e-6))
                dead_age[index].zero_()
            else:
                dead_age[index].add_(1)
        for index in range(prototypes.shape[0]):
            if dead_age[index] >= self.dead_prototype_patience:
                current = F.normalize(prototypes, dim=-1, eps=1e-6)
                least_represented = normalized.matmul(current.t()).max(
                    dim=-1
                ).values.argmin()
                prototypes[index].copy_(normalized[least_represented])
                dead_age[index].zero_()
                reinitialized += 1
        return shares, reinitialized

    @staticmethod
    def _bank_geometry(
        values: torch.Tensor, prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        if values.numel():
            assignments = F.normalize(values.float(), dim=-1, eps=1e-6).matmul(
                F.normalize(prototypes, dim=-1, eps=1e-6).t()
            ).argmax(dim=-1)
            counts = torch.bincount(
                assignments, minlength=prototypes.shape[0]
            ).float()
            shares = counts / counts.sum().clamp_min(1.0)
        else:
            shares = prototypes.new_zeros(prototypes.shape[0])
        pairwise_max = float("nan")
        if prototypes.shape[0] > 1:
            normalized = F.normalize(prototypes, dim=-1, eps=1e-6)
            pairwise = normalized.matmul(normalized.t()).masked_fill(
                torch.eye(
                    prototypes.shape[0],
                    dtype=torch.bool,
                    device=prototypes.device,
                ),
                -1.0,
            )
            pairwise_max = float(pairwise.max().item())
        return shares, pairwise_max

    @torch.no_grad()
    def finalize_epoch(self) -> Dict[str, object]:
        foreground = self._gather(
            self._local_foreground(), self.foreground_prototypes
        )
        background = self._gather(
            self._local_background(), self.background_prototypes
        )
        epoch = int(self.active_epoch.item())
        initialized = updated = reinitialized = 0
        enough = (
            foreground.shape[0] >= self.num_fg
            and background.shape[0] >= self.num_bg
        )
        if (
            not bool(self.prototype_ready.item())
            and epoch + 1 >= self.warmup_epochs
            and enough
        ):
            self.foreground_prototypes.copy_(
                self._spherical_kmeans(foreground, self.num_fg, 17)
            )
            self.background_prototypes.copy_(
                self._spherical_kmeans(background, self.num_bg, 1000003)
            )
            self.prototype_ready.fill_(1)
            self.foreground_dead_age.zero_()
            self.background_dead_age.zero_()
            self.prototype_update_count.add_(1)
            initialized = 1
        elif bool(self.prototype_ready.item()) and enough:
            _, fg_reinit = self._refresh_bank(
                foreground, self.foreground_prototypes, self.foreground_dead_age
            )
            _, bg_reinit = self._refresh_bank(
                background, self.background_prototypes, self.background_dead_age
            )
            reinitialized = fg_reinit + bg_reinit
            self.prototype_update_count.add_(1)
            updated = 1

        fg_shares, fg_pairwise = self._bank_geometry(
            foreground, self.foreground_prototypes
        )
        bg_shares, bg_pairwise = self._bank_geometry(
            background, self.background_prototypes
        )
        cross_max = float("nan")
        if bool(self.prototype_ready.item()):
            cross_max = float(
                self.normalized_foreground_prototypes()
                .matmul(self.normalized_background_prototypes().t())
                .max()
                .item()
            )
        self.active_epoch.add_(1)
        return {
            "prototype_ready": float(self.prototype_ready.item()),
            "prototype_initialization_event": float(initialized),
            "prototype_epoch_refresh_event": float(updated),
            "prototype_reinitialization_count": float(reinitialized),
            "prototype_update_count": float(self.prototype_update_count.item()),
            "dualproxy_active_epoch": float(epoch),
            "dualproxy_foreground_gathered": float(foreground.shape[0]),
            "dualproxy_background_gathered": float(background.shape[0]),
            "foreground_assignment_share": [float(x) for x in fg_shares.tolist()],
            "background_assignment_share": [float(x) for x in bg_shares.tolist()],
            "foreground_assignment_share_min": float(fg_shares.min().item()),
            "background_assignment_share_min": float(bg_shares.min().item()),
            "foreground_effective_prototypes": _effective_count(fg_shares),
            "background_effective_prototypes": _effective_count(bg_shares),
            "foreground_pairwise_similarity_max": fg_pairwise,
            "background_pairwise_similarity_max": bg_pairwise,
            "cross_bank_similarity_max": cross_max,
        }

    def fixed_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(DISCRIMINATIVE_DUAL_PROXY_VERSION.encode("utf-8"))
        return digest.hexdigest()
