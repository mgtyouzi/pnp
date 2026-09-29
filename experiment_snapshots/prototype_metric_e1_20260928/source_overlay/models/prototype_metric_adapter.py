import torch
from torch import nn
import torch.nn.functional as F


def _empty_indices(reference):
    return torch.empty(0, dtype=torch.long, device=reference.device)


def _stratified_score_sample(indices, scores, max_count):
    if max_count <= 0 or indices.numel() == 0:
        return _empty_indices(indices)
    if indices.numel() <= max_count:
        return indices

    order = torch.argsort(scores[indices])
    positions = torch.linspace(
        0,
        order.numel() - 1,
        steps=max_count,
        device=indices.device,
    ).round().long()
    return indices[order[positions]]


def select_proto_candidates(
    points,
    cell_scores,
    gt_points,
    matched_src,
    matched_tgt,
    max_positive=32,
    max_negative=64,
    positive_radius=15.0,
    negative_radius=30.0,
):
    """Select detached positive and hard-background indices for one image."""
    points_detached = points.detach().float()
    gt_points_detached = gt_points.detach().float()
    scores = cell_scores.detach().reshape(-1).float()
    query_count = points.shape[0]
    device = points.device

    matched_src = matched_src.detach().reshape(-1).to(device=device, dtype=torch.long)
    matched_tgt = matched_tgt.detach().reshape(-1).to(device=device, dtype=torch.long)
    pair_count = min(matched_src.numel(), matched_tgt.numel())
    matched_src = matched_src[:pair_count]
    matched_tgt = matched_tgt[:pair_count]

    if pair_count and gt_points_detached.shape[0]:
        valid_pairs = matched_tgt < gt_points_detached.shape[0]
        matched_src_valid = matched_src[valid_pairs]
        matched_tgt_valid = matched_tgt[valid_pairs]
        matched_distance = torch.norm(
            points_detached[matched_src_valid]
            - gt_points_detached[matched_tgt_valid],
            dim=-1,
        )
        valid_positive = matched_distance <= positive_radius
        positive_indices = matched_src_valid[valid_positive]
    else:
        positive_indices = _empty_indices(points)

    positive_indices = _stratified_score_sample(
        positive_indices, scores, max_positive
    )

    matched_mask = torch.zeros(query_count, dtype=torch.bool, device=device)
    if matched_src.numel():
        valid_src = matched_src[(matched_src >= 0) & (matched_src < query_count)]
        matched_mask[valid_src] = True
    unmatched_indices = torch.where(~matched_mask)[0]

    if gt_points_detached.shape[0]:
        nearest_gt_distance = torch.cdist(
            points_detached, gt_points_detached
        ).min(dim=1).values
    else:
        nearest_gt_distance = torch.full(
            (query_count,), float("inf"), device=device
        )
    far_mask = nearest_gt_distance > negative_radius
    negative_indices = unmatched_indices[far_mask[unmatched_indices]]

    if max_negative <= 0 or negative_indices.numel() == 0:
        negative_indices = _empty_indices(points)
    elif negative_indices.numel() > max_negative:
        order = torch.argsort(scores[negative_indices], descending=True)
        negative_indices = negative_indices[order[:max_negative]]

    return {
        "positive_indices": positive_indices,
        "negative_indices": negative_indices,
    }


def prototype_margin_loss(pos_margin, neg_margin, margin=0.1, beta=16.0):
    """Balanced soft margin loss for positive and negative candidates."""
    class_losses = []
    if pos_margin.numel():
        class_losses.append(
            F.softplus(beta * (margin - pos_margin)).mean() / beta
        )
    if neg_margin.numel():
        class_losses.append(
            F.softplus(beta * (margin + neg_margin)).mean() / beta
        )
    if not class_losses:
        return pos_margin.sum() * 0.0 + neg_margin.sum() * 0.0
    return torch.stack(class_losses).mean()


class PrototypeMetricAdapter(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.fg_proxy = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.bg_proxy = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def normalized_proxies(self):
        fg = F.normalize(self.fg_proxy, p=2, dim=-1)
        bg = F.normalize(self.bg_proxy, p=2, dim=-1)
        return fg, bg

    def zero_loss(self, features):
        parameter_zero = sum(
            (parameter.sum() * 0.0 for parameter in self.parameters()),
            features.sum() * 0.0,
        )
        return parameter_zero

    def forward(self, features):
        embedding = F.normalize(self.projector(features.float()), p=2, dim=-1)
        fg_proxy, bg_proxy = self.normalized_proxies()
        sim_fg = torch.matmul(embedding, fg_proxy.transpose(0, 1)).squeeze(-1)
        sim_bg = torch.matmul(embedding, bg_proxy.transpose(0, 1)).squeeze(-1)
        return {
            "embedding": embedding,
            "sim_fg": sim_fg,
            "sim_bg": sim_bg,
            "margin": sim_fg - sim_bg,
        }


def compute_image_proto_e1_loss(
    cls_features,
    cls_logits,
    points,
    gt_points,
    matched_src,
    matched_tgt,
    adapter,
    max_positive=32,
    max_negative=64,
    positive_radius=15.0,
    negative_radius=30.0,
    margin=0.1,
    beta=16.0,
):
    cell_scores = cls_logits.detach().softmax(dim=-1)[:, 0]
    selected = select_proto_candidates(
        points,
        cell_scores,
        gt_points,
        matched_src,
        matched_tgt,
        max_positive=max_positive,
        max_negative=max_negative,
        positive_radius=positive_radius,
        negative_radius=negative_radius,
    )
    pos_indices = selected["positive_indices"]
    neg_indices = selected["negative_indices"]

    if pos_indices.numel() + neg_indices.numel() == 0:
        zero = adapter.zero_loss(cls_features)
        return zero, {
            "loss_proto_raw": 0.0,
            "num_positive": 0.0,
            "num_negative": 0.0,
            "proxy_cosine": float(
                torch.sum(adapter.normalized_proxies()[0]
                          * adapter.normalized_proxies()[1]).detach().item()
            ),
            "pos_margin_mean": float("nan"),
            "pos_margin_std": float("nan"),
            "neg_margin_mean": float("nan"),
            "neg_margin_std": float("nan"),
            "proto_pos_acc": float("nan"),
            "proto_neg_acc": float("nan"),
        }

    chosen = torch.cat((pos_indices, neg_indices), dim=0)
    proto_output = adapter(cls_features[chosen])
    pos_count = int(pos_indices.numel())
    pos_margin = proto_output["margin"][:pos_count]
    neg_margin = proto_output["margin"][pos_count:]
    loss = prototype_margin_loss(pos_margin, neg_margin, margin, beta)

    with torch.no_grad():
        fg_proxy, bg_proxy = adapter.normalized_proxies()
        diagnostics = {
            "loss_proto_raw": float(loss.detach().item()),
            "num_positive": float(pos_count),
            "num_negative": float(neg_indices.numel()),
            "proxy_cosine": float(torch.sum(fg_proxy * bg_proxy).item()),
            "pos_margin_mean": (
                float(pos_margin.mean().item()) if pos_count else float("nan")
            ),
            "pos_margin_std": (
                float(pos_margin.std(unbiased=False).item())
                if pos_count else float("nan")
            ),
            "neg_margin_mean": (
                float(neg_margin.mean().item())
                if neg_margin.numel() else float("nan")
            ),
            "neg_margin_std": (
                float(neg_margin.std(unbiased=False).item())
                if neg_margin.numel() else float("nan")
            ),
            "proto_pos_acc": (
                float((pos_margin > 0).float().mean().item())
                if pos_count else float("nan")
            ),
            "proto_neg_acc": (
                float((neg_margin < 0).float().mean().item())
                if neg_margin.numel() else float("nan")
            ),
        }
    return loss, diagnostics
