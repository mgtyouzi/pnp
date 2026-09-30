import math

import torch
from torch import nn
import torch.nn.functional as F


def _stratified_indices(indices, scores, count):
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    order = torch.argsort(scores[indices])
    positions = torch.linspace(
        0, order.numel() - 1, steps=count, device=indices.device
    ).round().long()
    return indices[order[positions]]


def _mixed_hard_stratified_negatives(indices, scores, count):
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    hard_count = max(1, count // 2)
    hard = indices[torch.argsort(scores[indices], descending=True)[:hard_count]]
    spread = _stratified_indices(indices, scores, count)
    chosen = torch.cat((hard, spread)).unique(sorted=False)
    if chosen.numel() < count:
        order = indices[torch.argsort(scores[indices], descending=True)]
        used = torch.zeros_like(scores, dtype=torch.bool)
        used[chosen] = True
        used = used[order]
        chosen = torch.cat((chosen, order[~used][:count - chosen.numel()]))
    return chosen[:count]


def select_e2_candidates(
    points,
    cell_scores,
    gt_points,
    matched_src,
    matched_tgt,
    max_positive=32,
    max_negative=64,
    positive_radius=15.0,
    negative_radius=30.0,
    negative_score_floor=0.1,
):
    """Select matched positives and unmatched far-background queries per image."""
    points = points.detach().float()
    scores = cell_scores.detach().reshape(-1).float()
    gt_points = gt_points.detach().float().reshape(-1, 2)
    query_count = points.shape[0]
    device = points.device
    matched_src = matched_src.detach().reshape(-1).to(device=device, dtype=torch.long)
    matched_tgt = matched_tgt.detach().reshape(-1).to(device=device, dtype=torch.long)
    valid = (matched_src >= 0) & (matched_src < query_count)
    valid &= (matched_tgt >= 0) & (matched_tgt < gt_points.shape[0])
    src, tgt = matched_src[valid], matched_tgt[valid]

    matched_mask = torch.zeros(query_count, dtype=torch.bool, device=device)
    matched_mask[src] = True
    if src.numel():
        distance = torch.linalg.vector_norm(points[src] - gt_points[tgt], dim=-1)
        positive = src[distance <= positive_radius]
    else:
        positive = torch.empty(0, dtype=torch.long, device=device)
    positive = _stratified_indices(positive, scores, max_positive)

    unmatched = torch.where(~matched_mask)[0]
    if gt_points.numel():
        nearest = torch.cdist(points, gt_points).min(dim=1).values
    else:
        nearest = torch.full((query_count,), float("inf"), device=device)
    far = unmatched[(nearest[unmatched] > negative_radius)
                    & (scores[unmatched] >= negative_score_floor)]
    negative = _mixed_hard_stratified_negatives(far, scores, max_negative)
    return {"positive_indices": positive, "negative_indices": negative}


def balanced_prototype_margin_loss(pos_margin, neg_margin, margin=0.1, beta=16.0):
    terms = []
    if pos_margin.numel():
        terms.append(F.softplus(beta * (margin - pos_margin)).mean() / beta)
    if neg_margin.numel():
        terms.append(F.softplus(beta * (margin + neg_margin)).mean() / beta)
    if not terms:
        return pos_margin.sum() * 0.0 + neg_margin.sum() * 0.0
    return torch.stack(terms).mean()


class E2PrototypeResidualAdapter(nn.Module):
    """Detached-feature metric branch with a bounded binary-logit residual."""

    def __init__(
        self,
        in_dim=256,
        hidden_dim=128,
        embed_dim=64,
        alpha_init=0.05,
        alpha_max=0.5,
        r_max=1.0,
    ):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.fg_proxy = nn.Parameter(F.normalize(torch.randn(1, embed_dim), dim=-1))
        self.bg_proxy = nn.Parameter(F.normalize(torch.randn(1, embed_dim), dim=-1))
        self.alpha_max = float(alpha_max)
        self.r_max = float(r_max)
        initial_logit = math.log(alpha_init / (alpha_max - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def normalized_proxies(self):
        return (
            F.normalize(self.fg_proxy, p=2, dim=-1),
            F.normalize(self.bg_proxy, p=2, dim=-1),
        )

    def zero_loss(self, reference):
        zero = reference.sum() * 0.0
        for parameter in self.parameters():
            zero = zero + parameter.sum() * 0.0
        return zero

    def forward(self, features, base_logits, residual_enabled=True):
        if base_logits.shape[-1] != 2:
            raise ValueError("E2 residual adapter requires one foreground class plus background")
        embedding = F.normalize(self.projector(features.detach().float()), dim=-1, eps=1e-6)
        fg_proxy, bg_proxy = self.normalized_proxies()
        sim_fg = torch.matmul(embedding, fg_proxy.transpose(0, 1)).squeeze(-1)
        sim_bg = torch.matmul(embedding, bg_proxy.transpose(0, 1)).squeeze(-1)
        margin = sim_fg - sim_bg
        cell_probability = base_logits.detach().softmax(dim=-1)[..., 0]
        gate = (4.0 * cell_probability * (1.0 - cell_probability)).clamp(0.0, 1.0)
        raw_residual = self.alpha * gate * margin
        residual = self.r_max * torch.tanh(raw_residual / self.r_max)
        if not residual_enabled:
            residual = torch.zeros_like(residual)
        adjusted = self._apply_residual(base_logits, residual)
        calibration_logits = self._apply_residual(base_logits.detach(), residual)
        return {
            "embedding": embedding,
            "sim_fg": sim_fg,
            "sim_bg": sim_bg,
            "margin": margin,
            "gate": gate,
            "residual": residual,
            "cls_logits": adjusted,
            "calibration_logits": calibration_logits,
            "alpha": self.alpha,
        }

    @staticmethod
    def _apply_residual(logits, residual):
        half = 0.5 * residual
        return torch.stack(
            (logits[..., 0] + half, logits[..., 1] - half), dim=-1
        )


def compute_e2_image_loss(
    outputs,
    points,
    gt_points,
    matched_src,
    matched_tgt,
    adapter,
    args,
    geometry_enabled,
    calibration_enabled,
    batch_index=0,
):
    """Train the metric branch and/or its final-logit calibration per image."""
    zero = adapter.zero_loss(outputs["cls_features"][batch_index])
    if not geometry_enabled and not calibration_enabled:
        fg_proxy, bg_proxy = adapter.normalized_proxies()
        return zero, {
            "positive": 0.0, "negative": 0.0, "geometry": 0.0,
            "calibration": 0.0, "margin_gap": float("nan"),
            "alpha": float(adapter.alpha.detach().item()),
            "fg_bg_cosine": float((fg_proxy * bg_proxy).sum().detach().item()),
        }

    scores = outputs["cls_logits_base"].detach().softmax(dim=-1)[batch_index, :, 0]
    selected = select_e2_candidates(
        points, scores, gt_points, matched_src, matched_tgt,
        max_positive=args.e2_max_positive,
        max_negative=args.e2_max_negative,
        positive_radius=args.e2_positive_radius,
        negative_radius=args.e2_negative_radius,
        negative_score_floor=args.e2_negative_score_floor,
    )
    pos = selected["positive_indices"]
    neg = selected["negative_indices"]
    margin = outputs["e2_margin"][batch_index]
    pos_margin, neg_margin = margin[pos], margin[neg]

    geometry = zero
    if geometry_enabled and (pos.numel() or neg.numel()):
        geometry = balanced_prototype_margin_loss(
            pos_margin, neg_margin,
            margin=args.e2_metric_margin,
            beta=args.e2_metric_beta,
        )

    calibration = zero
    if calibration_enabled and (pos.numel() or neg.numel()):
        chosen = torch.cat((pos, neg))
        labels = torch.cat((
            torch.zeros(pos.numel(), dtype=torch.long, device=points.device),
            torch.ones(neg.numel(), dtype=torch.long, device=points.device),
        ))
        per_query = F.cross_entropy(
            outputs["e2_calibration_logits"][batch_index, chosen], labels, reduction="none"
        )
        terms = []
        if pos.numel():
            terms.append(per_query[:pos.numel()].mean())
        if neg.numel():
            terms.append(per_query[pos.numel():].mean())
        calibration = torch.stack(terms).mean()

    loss = calibration + args.e2_metric_loss_weight * geometry
    gap = (
        float(pos_margin.mean().detach().item() - neg_margin.mean().detach().item())
        if pos.numel() and neg.numel() else float("nan")
    )
    diagnostics = {
        "positive": float(pos.numel()),
        "negative": float(neg.numel()),
        "geometry": float(geometry.detach().item()),
        "calibration": float(calibration.detach().item()),
        "margin_gap": gap,
        "alpha": float(adapter.alpha.detach().item()),
        "fg_bg_cosine": float((adapter.normalized_proxies()[0] * adapter.normalized_proxies()[1]).sum().detach().item()),
    }
    return loss, diagnostics
