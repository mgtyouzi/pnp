# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import copy

import numpy as np
import torch.nn.functional as F

from torch import nn
from models.backbone import build_backbone
from models.prototype_metric_adapter import PrototypeMetricAdapter
from models.prototype_residual_adapter_e2 import E2PrototypeResidualAdapter


class AnchorPoints(nn.Module):
    def __init__(self, row=2, col=2, grid_scale=(32, 32)):
        super(AnchorPoints, self).__init__()
        x_space = grid_scale[0] / row
        y_space = grid_scale[1] / col
        self.deltas = np.array(
            [
                [-x_space, -y_space],
                [x_space, -y_space],
                [0, 0],
                [-x_space, y_space],
                [x_space, y_space]
            ]
        ) / 2

        self.grid_scale = np.array(grid_scale)

    def forward(self, images):
        bs, _, h, w = images.shape
        centers = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.grid_scale[0])) + 0.5,
                np.arange(np.ceil(h / self.grid_scale[1])) + 0.5),
            -1) * self.grid_scale

        anchors = np.expand_dims(centers, 2) + self.deltas
        anchors = torch.from_numpy(anchors).float().to(images.device)

        return anchors.flatten(0, 2).repeat(bs, 1, 1)


class DETR(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_dim,
        num_classes,
        row,
        col,
        use_proto_e1=False,
        proto_hidden_dim=128,
        proto_embed_dim=64,
        use_proto_e2=False,
        e2_hidden_dim=128,
        e2_embed_dim=64,
        e2_alpha_init=0.05,
        e2_alpha_max=0.5,
        e2_residual_max=1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(row, col)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_levels = self.backbone.neck.num_outs
        self.strides = [2 ** (i + 1) for i in range(self.num_levels)]

        # deformable roi pooling
        self.deformable_mlps = _get_clones(MLP(hidden_dim, hidden_dim, 2, 2), self.num_levels)
        for mlp in self.deformable_mlps:
            nn.init.constant_(mlp.layers[-1].bias.data, 0)
            nn.init.constant_(mlp.layers[-1].weight.data, 0)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 3)
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)

        self.loc_aggr = nn.Sequential(nn.Linear(hidden_dim * self.num_levels, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, self.num_levels))
        self.cls_aggr = nn.Sequential(nn.Linear(hidden_dim * self.num_levels, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, self.num_levels))
        self.use_proto_e1 = bool(use_proto_e1)
        self.proto_metric_adapter = (
            PrototypeMetricAdapter(
                in_dim=hidden_dim,
                hidden_dim=proto_hidden_dim,
                embed_dim=proto_embed_dim,
            )
            if self.use_proto_e1 else None
        )
        self.use_proto_e2 = bool(use_proto_e2)
        if self.use_proto_e1 and self.use_proto_e2:
            raise ValueError("E1 and E2 prototype adapters are mutually exclusive")
        if self.use_proto_e2 and num_classes != 1:
            raise ValueError("E2 currently supports exactly one foreground class")
        self.prototype_residual_adapter_e2 = (
            E2PrototypeResidualAdapter(
                in_dim=hidden_dim,
                hidden_dim=e2_hidden_dim,
                embed_dim=e2_embed_dim,
                alpha_init=e2_alpha_init,
                alpha_max=e2_alpha_max,
                r_max=e2_residual_max,
            )
            if self.use_proto_e2 else None
        )
        self.register_buffer('e2_residual_enabled_state', torch.tensor(True, dtype=torch.bool))

    @property
    def e2_residual_enabled(self):
        return bool(self.e2_residual_enabled_state.item())

    @e2_residual_enabled.setter
    def e2_residual_enabled(self, enabled):
        self.e2_residual_enabled_state.fill_(bool(enabled))

    def set_e2_residual_enabled(self, enabled):
        self.e2_residual_enabled = bool(enabled)

    def forward(self, images):
        anchors = self.get_aps(images)
        features = self.backbone(images)

        reg_features, cls_features, reg_attn, cls_attn = self.extract_features(features, anchors)
        pnt_coords = self.reg_head(reg_features) + anchors
        cls_logits = self.cls_head(cls_features)

        # outputs = {'pnt_coords': pnt_coords, 'cls_logits': cls_logits}
        outputs = {'pnt_coords': pnt_coords, 'cls_logits': cls_logits, 'reg_attn': reg_attn, 'cls_attn': cls_attn}
        if self.training and self.proto_metric_adapter is not None:
            outputs['cls_features'] = cls_features
        if self.prototype_residual_adapter_e2 is not None:
            e2 = self.prototype_residual_adapter_e2(
                cls_features,
                cls_logits,
                residual_enabled=self.e2_residual_enabled,
            )
            outputs.update({
                'cls_logits_base': cls_logits,
                'cls_logits': e2['cls_logits'],
                'e2_calibration_logits': e2['calibration_logits'],
                'cls_features': cls_features,
                'e2_embedding': e2['embedding'],
                'e2_sim_fg': e2['sim_fg'],
                'e2_sim_bg': e2['sim_bg'],
                'e2_margin': e2['margin'],
                'e2_gate': e2['gate'],
                'e2_residual': e2['residual'],
            })
        return outputs

    def extract_features(self, features, points, align_corners=True):
        roi_features = torch.zeros(self.num_levels, *points.shape[:2], self.hidden_dim).cuda(points.device)
        for i, stride in enumerate(self.strides):
            h, w = features[i].shape[2:]
            scale = torch.tensor([w, h], dtype=torch.float, device=points.device)
            grid = (2.0 * points / stride / scale - 1.0).unsqueeze(2)  # for alignment

            pre_roi_features = F.grid_sample(features[i], grid, align_corners=align_corners).squeeze(-1).permute(0, 2,
                                                                                                                 1)
            grid = (2.0 * (points + self.deformable_mlps[i](pre_roi_features)) / stride / scale - 1.0).unsqueeze(2)

            roi_features[i] = F.grid_sample(features[i], grid, align_corners=align_corners).squeeze(-1).permute(0, 2, 1)

        roi_features = roi_features.permute(1, 2, 0, 3)
        attn_features = roi_features.flatten(2)

        reg_attn = F.softmax(self.loc_aggr(attn_features), dim=-1).unsqueeze(-1)
        reg_features = (reg_attn * roi_features).sum(dim=2)

        cls_attn = F.softmax(self.cls_aggr(attn_features), dim=-1).unsqueeze(-1)
        cls_features = (cls_attn * roi_features).sum(dim=2)

        return reg_features, cls_features, reg_attn, cls_attn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_model(args):
    backbone = build_backbone(args)

    model = DETR(
        backbone,
        row=args.row,
        col=args.col,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        use_proto_e1=bool(getattr(args, 'use_proto_e1', False)),
        proto_hidden_dim=getattr(args, 'proto_hidden_dim', 128),
        proto_embed_dim=getattr(args, 'proto_embed_dim', 64),
        use_proto_e2=bool(getattr(args, 'use_proto_e2', False)),
        e2_hidden_dim=getattr(args, 'e2_hidden_dim', 128),
        e2_embed_dim=getattr(args, 'e2_embed_dim', 64),
        e2_alpha_init=getattr(args, 'e2_alpha_init', 0.05),
        e2_alpha_max=getattr(args, 'e2_alpha_max', 0.5),
        e2_residual_max=getattr(args, 'e2_residual_max', 1.0),
    )

    return model
