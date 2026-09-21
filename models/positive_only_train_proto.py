import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


POSITIVE_ONLY_TRAIN_PROTO_VERSION = "positive_only_train_proto_v2_20260904"


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


class PositiveOnlyTrainPrototype(nn.Module):
    """Training-only cell prototypes with no background bank or score fusion."""

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_prototypes: int = 4,
        warmup_epochs: int = 2,
        temperature: float = 0.1,
        positive_margin: float = 0.4,
        negative_margin: float = 0.2,
        diversity_margin: float = 0.5,
        pair_weight: float = 0.01,
        positive_weight: float = 0.01,
        negative_weight: float = 0.005,
        balance_weight: float = 0.001,
        diversity_weight: float = 0.001,
        input_gradient_scale: float = 0.05,
        background_radius: float = 30.0,
        max_hard_background_per_image: int = 32,
        max_random_background_per_image: int = 16,
        support_reservoir_size: int = 8192,
        kmeans_iterations: int = 20,
        sampling_seed: int = 0,
        hard_positive_fraction: float = 1.0,
        max_hard_positive_per_image: int = 0,
        detach_support_input: bool = False,
    ) -> None:
        super().__init__()
        if min(feat_dim, hidden_dim, embedding_dim, num_prototypes) <= 0:
            raise ValueError("prototype dimensions and counts must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if background_radius <= 0:
            raise ValueError("background_radius must be positive")
        if not 0 <= input_gradient_scale <= 1:
            raise ValueError("input_gradient_scale must be in [0, 1]")
        if not 0 < hard_positive_fraction <= 1:
            raise ValueError("hard_positive_fraction must be in (0, 1]")
        if max_hard_positive_per_image < 0:
            raise ValueError("max_hard_positive_per_image must be non-negative")

        self.num_prototypes = int(num_prototypes)
        self.embedding_dim = int(embedding_dim)
        self.warmup_epochs = int(warmup_epochs)
        self.temperature = float(temperature)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.diversity_margin = float(diversity_margin)
        self.pair_weight = float(pair_weight)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.balance_weight = float(balance_weight)
        self.diversity_weight = float(diversity_weight)
        self.input_gradient_scale = float(input_gradient_scale)
        self.background_radius = float(background_radius)
        self.max_hard_background_per_image = int(max_hard_background_per_image)
        self.max_random_background_per_image = int(max_random_background_per_image)
        self.support_reservoir_size = int(support_reservoir_size)
        self.kmeans_iterations = int(kmeans_iterations)
        self.sampling_seed = int(sampling_seed)
        self.hard_positive_fraction = float(hard_positive_fraction)
        self.max_hard_positive_per_image = int(max_hard_positive_per_image)
        self.detach_support_input = bool(detach_support_input)
        self.current_epoch = 0

        self.projector = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sampling_seed + 104729)
        centers = torch.randn(
            self.num_prototypes, self.embedding_dim, generator=generator
        )
        self.foreground_prototypes = nn.Parameter(F.normalize(centers, dim=-1))
        self.register_buffer("prototype_ready", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("sampling_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "support_reservoir",
            torch.zeros(self.support_reservoir_size, self.embedding_dim),
        )
        self.register_buffer(
            "support_priorities", torch.full((self.support_reservoir_size,), -1.0)
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
        scaled = scale_gradient(values, scale)
        return F.normalize(self.projector(scaled.float()), dim=-1, eps=1e-6)

    def foreground_scores(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        similarities = embeddings.matmul(self.normalized_prototypes().t())
        assignments = F.softmax(similarities / self.temperature, dim=-1)
        score = self.temperature * (
            torch.logsumexp(similarities / self.temperature, dim=-1)
            - math.log(self.num_prototypes)
        )
        return score, assignments

    def _cache_initial_support(self, values: torch.Tensor) -> None:
        if bool(self.prototype_ready.item()) or values.numel() == 0:
            return
        values = values.detach().reshape(-1, self.embedding_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.sampling_seed + 65_537 * (int(self.sampling_step.item()) + 1)
        )
        priorities = torch.rand(
            values.shape[0], generator=generator, device="cpu"
        ).to(values.device)
        count = int(self.support_count.item())
        existing_values = self.support_reservoir[:count]
        existing_priorities = self.support_priorities[:count]
        combined_values = torch.cat([existing_values, values], dim=0)
        combined_priorities = torch.cat([existing_priorities, priorities], dim=0)
        keep = min(self.support_reservoir_size, int(combined_values.shape[0]))
        selected = torch.topk(combined_priorities, keep, sorted=False).indices
        self.support_reservoir[:keep].copy_(combined_values[selected])
        self.support_priorities[:keep].copy_(combined_priorities[selected])
        self.support_count.fill_(keep)

    def _spherical_kmeans(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1, eps=1e-6)
        if values.shape[0] < self.num_prototypes:
            raise RuntimeError("not enough exact-GT supports to initialize prototypes")
        first = self.sampling_seed % int(values.shape[0])
        centers = [values[first]]
        for _ in range(1, self.num_prototypes):
            existing = torch.stack(centers, dim=0)
            nearest_similarity = values.matmul(existing.t()).max(dim=1).values
            centers.append(values[nearest_similarity.argmin()])
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

    def forward(
        self,
        cls_features: torch.Tensor,
        raw_logits: torch.Tensor,
        support_features: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self._embed(cls_features)
        if support_features is None:
            support_embeddings = embeddings.new_zeros(
                embeddings.shape[0], 0, embeddings.shape[-1]
            )
        else:
            support_embeddings = self._embed(
                support_features,
                input_gradient_scale=0.0 if self.detach_support_input else None,
            )
        foreground, _ = self.foreground_scores(embeddings)
        return {
            "embeddings": embeddings,
            "support_embeddings": support_embeddings,
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

    def sample_candidates(
        self,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        sampled = {
            "positive_indices": [],
            "positive_target_indices": [],
            "ignored_near_indices": [],
            "hard_background_indices": [],
            "random_background_indices": [],
            "matched_positive_count": [],
            "selected_positive_probabilities": [],
            "unselected_positive_probabilities": [],
        }
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            device = points.device
            source_indices = source_indices.to(device=device, dtype=torch.long)
            target_indices = target_indices.to(device=device, dtype=torch.long)
            gt_points = targets["gt_points"][batch_index].to(
                device=device, dtype=torch.float32
            )
            matched = torch.zeros(points.shape[1], dtype=torch.bool, device=device)
            matched[source_indices] = True
            if gt_points.numel():
                nearest = torch.cdist(
                    points[batch_index].float(), gt_points.float(), p=2
                ).min(dim=1).values
            else:
                nearest = points.new_full(
                    (points.shape[1],), float("inf"), dtype=torch.float32
                )
            ignored_near = torch.where(
                (~matched) & (nearest <= self.background_radius)
            )[0]
            eligible = torch.where(
                (~matched) & (nearest > self.background_radius)
            )[0]
            probability = raw_logits[batch_index].softmax(dim=-1)[:, 0]
            matched_probability = probability[source_indices]
            selected_count = int(math.ceil(
                int(source_indices.numel()) * self.hard_positive_fraction
            ))
            if self.max_hard_positive_per_image > 0:
                selected_count = min(
                    selected_count, self.max_hard_positive_per_image
                )
            selected_count = min(selected_count, int(source_indices.numel()))
            if selected_count < int(source_indices.numel()):
                selected_positions = torch.topk(
                    -matched_probability, selected_count
                ).indices
                selected_mask = torch.zeros(
                    source_indices.numel(), dtype=torch.bool, device=device
                )
                selected_mask[selected_positions] = True
                unselected_probability = matched_probability[~selected_mask]
                source_indices = source_indices[selected_positions]
                target_indices = target_indices[selected_positions]
                selected_probability = matched_probability[selected_positions]
            else:
                selected_probability = matched_probability
                unselected_probability = matched_probability[:0]
            hard_count = min(self.max_hard_background_per_image, int(eligible.numel()))
            hard = (
                eligible[torch.topk(probability[eligible], hard_count).indices]
                if hard_count
                else eligible[:0]
            )
            hard_mask = torch.zeros(points.shape[1], dtype=torch.bool, device=device)
            hard_mask[hard] = True
            remaining = eligible[~hard_mask[eligible]]
            random_background = self._sample_without_global_rng(
                remaining,
                min(self.max_random_background_per_image, int(remaining.numel())),
                batch_index,
            )
            sampled["positive_indices"].append(source_indices)
            sampled["positive_target_indices"].append(target_indices)
            sampled["ignored_near_indices"].append(ignored_near)
            sampled["hard_background_indices"].append(hard)
            sampled["random_background_indices"].append(random_background)
            sampled["matched_positive_count"].append(int(matched_probability.numel()))
            sampled["selected_positive_probabilities"].append(selected_probability)
            sampled["unselected_positive_probabilities"].append(
                unselected_probability
            )
        self.sampling_step.add_(1)
        return sampled

    @staticmethod
    def _cat_or_empty(values: List[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        values = [value for value in values if value.numel()]
        return torch.cat(values, dim=0) if values else reference.reshape(-1, reference.shape[-1])[:0]

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_valid_mask: torch.Tensor,
        points: torch.Tensor,
        raw_logits: torch.Tensor,
        targets: Dict,
        indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        anchor_points: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del anchor_points
        sampled = self.sample_candidates(points, raw_logits, targets, indices)
        query_values = []
        support_values = []
        hard_values = []
        random_values = []
        for batch_index in range(embeddings.shape[0]):
            query_index = sampled["positive_indices"][batch_index]
            target_index = sampled["positive_target_indices"][batch_index]
            valid_pair = target_index < support_embeddings.shape[1]
            if support_valid_mask.numel() and target_index.numel():
                valid_pair = valid_pair & support_valid_mask[batch_index, target_index]
            query_values.append(embeddings[batch_index, query_index[valid_pair]])
            support_values.append(
                support_embeddings[batch_index, target_index[valid_pair]]
            )
            hard_values.append(
                embeddings[batch_index, sampled["hard_background_indices"][batch_index]]
            )
            random_values.append(
                embeddings[batch_index, sampled["random_background_indices"][batch_index]]
            )

        query = self._cat_or_empty(query_values, embeddings)
        support = self._cat_or_empty(support_values, embeddings)
        hard_background = self._cat_or_empty(hard_values, embeddings)
        random_background = self._cat_or_empty(random_values, embeddings)
        background = self._cat_or_empty(
            [hard_background, random_background], embeddings
        )
        if support_valid_mask.numel():
            all_support = support_embeddings[support_valid_mask.bool()]
        else:
            all_support = support_embeddings.reshape(
                -1, support_embeddings.shape[-1]
            )[:0]
        zero = embeddings.sum() * 0.0

        self._cache_initial_support(all_support)

        positive_values = self._cat_or_empty([all_support, query], embeddings)
        if positive_values.numel() and bool(self.prototype_ready.item()):
            positive_scores, positive_assignments = self.foreground_scores(positive_values)
            positive_loss = (
                F.softplus(
                    (self.positive_margin - positive_scores) / self.temperature
                ).mean()
                * self.temperature
            )
            shares = positive_assignments.mean(dim=0)
            balance_loss = (
                shares * (shares.clamp_min(1e-8) * self.num_prototypes).log()
            ).sum()
        else:
            positive_scores = embeddings.new_zeros(0)
            shares = embeddings.new_zeros(self.num_prototypes)
            positive_loss = zero
            balance_loss = zero

        if background.numel() and bool(self.prototype_ready.item()):
            background_scores, _ = self.foreground_scores(background)
            negative_loss = (
                F.softplus(
                    (background_scores - self.negative_margin) / self.temperature
                ).mean()
                * self.temperature
            )
        else:
            background_scores = embeddings.new_zeros(0)
            negative_loss = zero

        pair_loss = (
            (1.0 - F.cosine_similarity(query, support.detach(), dim=-1)).mean()
            if query.numel()
            else zero
        )
        centers = self.normalized_prototypes()
        pairwise = centers.matmul(centers.t())
        upper = torch.triu_indices(
            self.num_prototypes, self.num_prototypes, offset=1, device=centers.device
        )
        diversity_loss = (
            F.relu(pairwise[upper[0], upper[1]] - self.diversity_margin).mean()
            if upper.shape[1] and bool(self.prototype_ready.item())
            else zero
        )
        loss = (
            self.pair_weight * pair_loss
            + self.positive_weight * positive_loss
            + self.negative_weight * negative_loss
            + self.balance_weight * balance_loss
            + self.diversity_weight * diversity_loss
        )
        diagnostics = {
            "potp_loss_raw": float(loss.detach().item()),
            "potp_loss_pair": float(pair_loss.detach().item()),
            "potp_loss_positive": float(positive_loss.detach().item()),
            "potp_loss_negative": float(negative_loss.detach().item()),
            "potp_loss_balance": float(balance_loss.detach().item()),
            "potp_loss_diversity": float(diversity_loss.detach().item()),
            "potp_positive_queries": float(query.shape[0]),
            "potp_matched_positive_total": float(
                sum(sampled["matched_positive_count"])
            ),
            "potp_unselected_positive": float(
                sum(value.numel() for value in sampled["unselected_positive_probabilities"])
            ),
            "potp_selected_positive_probability": float(
                torch.cat(sampled["selected_positive_probabilities"]).mean().item()
            ) if any(value.numel() for value in sampled["selected_positive_probabilities"]) else float("nan"),
            "potp_unselected_positive_probability": float(
                torch.cat(sampled["unselected_positive_probabilities"]).mean().item()
            ) if any(value.numel() for value in sampled["unselected_positive_probabilities"]) else float("nan"),
            "potp_gt_support": float(all_support.shape[0]),
            "potp_hard_background": float(hard_background.shape[0]),
            "potp_random_background": float(random_background.shape[0]),
            "potp_ignored_near": float(sum(x.numel() for x in sampled["ignored_near_indices"])),
            "potp_positive_score": float(positive_scores.mean().item()) if positive_scores.numel() else float("nan"),
            "potp_background_score": float(background_scores.mean().item()) if background_scores.numel() else float("nan"),
            "potp_score_gap": float(positive_scores.mean().item() - background_scores.mean().item()) if positive_scores.numel() and background_scores.numel() else float("nan"),
            "potp_pair_cosine": float((1.0 - pair_loss).detach().item()) if query.numel() else float("nan"),
            "potp_effective_prototypes": _effective_count(shares.detach()),
            "potp_assignment_share_min": float(shares.min().item()) if shares.numel() else 0.0,
            "potp_shared_gradient_scale": self.shared_gradient_scale(),
            "potp_support_input_gradient_scale": (
                0.0 if self.detach_support_input else self.shared_gradient_scale()
            ),
            "prototype_ready": float(self.prototype_ready.item()),
            "potp_initial_support_cached": float(self.support_count.item()),
        }
        return loss, diagnostics

    def begin_epoch(self) -> None:
        return None

    def maybe_refresh_prototypes(self) -> Dict[str, float]:
        return {}

    def finalize_epoch(self) -> Dict[str, float]:
        initialization_event = 0.0
        if not bool(self.prototype_ready.item()):
            local = self.support_reservoir[: int(self.support_count.item())].detach().cpu()
            if dist.is_available() and dist.is_initialized():
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, local)
                rank = dist.get_rank()
            else:
                gathered = [local]
                rank = 0
            if rank == 0:
                values = torch.cat(gathered, dim=0)
                centers = self._spherical_kmeans(values).to(
                    device=self.foreground_prototypes.device
                )
            else:
                centers = torch.zeros_like(self.foreground_prototypes)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(centers, src=0)
            with torch.no_grad():
                self.foreground_prototypes.copy_(centers)
                self.prototype_ready.fill_(1)
            initialization_event = 1.0
        centers = self.normalized_prototypes().detach()
        pairwise = centers.matmul(centers.t())
        if self.num_prototypes > 1:
            pairwise.fill_diagonal_(-1.0)
            pairwise_max = float(pairwise.max().item())
        else:
            pairwise_max = float("nan")
        return {
            "potp_foreground_pairwise_similarity_max": pairwise_max,
            "potp_shared_gradient_scale": self.shared_gradient_scale(),
            "potp_has_background_prototypes": 0.0,
            "potp_inference_fusion": 0.0,
            "potp_initialization_event": initialization_event,
            "potp_initial_support_cached": float(self.support_count.item()),
        }
