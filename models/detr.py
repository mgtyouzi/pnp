# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import copy

import numpy as np
import torch.nn.functional as F

from torch import nn
from models.backbone import build_backbone
from models.p2p_prototype import CandidatePrototypeBank
from models.frozen_teacher_fg_prototype import FrozenTeacherForegroundPrototype
from models.frozen_supervised_metric_prototype import (
    FrozenSupervisedMetricPrototype,
)
from models.gt_foreground_proxy import (
    GT_FOREGROUND_PROXY_VERSION,
    GTForegroundProxy,
)
from models.discriminative_dual_proxy import (
    DISCRIMINATIVE_DUAL_PROXY_VERSION,
    DiscriminativeDualProxy,
)
from models.source_supervised_candidate_proto import (
    SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION,
    SourceSupervisedCandidatePrototype,
)
from models.positive_only_train_proto import (
    POSITIVE_ONLY_TRAIN_PROTO_VERSION,
    PositiveOnlyTrainPrototype,
)
from models.prototype_guided_ranking import (
    PROTOTYPE_GUIDED_RANKING_VERSION,
    PrototypeGuidedRanking,
)
from models.prototype_reliability_rescue import (
    PROTOTYPE_RELIABILITY_RESCUE_VERSION,
    PrototypeReliabilityRescue,
)
from models.prototype_local_soft_positive import (
    PROTOTYPE_LOCAL_SOFT_POSITIVE_VERSION,
    PrototypeLocalSoftPositive,
)


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
    def __init__(self, backbone, hidden_dim, num_classes, row, col, args=None):
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

        self.prototype_head = None
        self.prototype_mode = "disabled"
        self.prototype_metadata = {}
        self.prototype_diagnostic_eval = False
        self.prototype_inference_fusion = False
        self.prototype_debug_enabled = False
        self._prototype_debug_tensors = None
        if args is not None and bool(getattr(args, 'proto_enable', False)):
            if num_classes != 1:
                raise ValueError('P2P prototype branch currently supports num_classes=1 only')
            self.prototype_mode = str(getattr(args, 'proto_mode', 'legacy_online'))
            if self.prototype_mode not in {
                'legacy_online',
                'frozen_teacher_fg',
                'frozen_supervised_metric',
                'gt_foreground_proxy',
                'discriminative_dual_proxy',
                'source_supervised_candidate_proto',
                'positive_only_train_proto',
                'prototype_guided_ranking',
                'prototype_reliability_rescue',
                'prototype_local_soft_positive',
            }:
                raise ValueError(f'unsupported proto_mode: {self.prototype_mode}')
            self.prototype_inference_fusion = bool(
                getattr(args, 'proto_inference_fusion', 0)
            )
            self.prototype_debug_enabled = int(
                getattr(args, 'proto_debug_interval', 0)
            ) > 0
            if self.prototype_mode in {
                'prototype_reliability_rescue',
                'prototype_local_soft_positive',
            }:
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is disabled for '
                        f'{self.prototype_mode}'
                    )
                prototype_class = (
                    PrototypeLocalSoftPositive
                    if self.prototype_mode == 'prototype_local_soft_positive'
                    else PrototypeReliabilityRescue
                )
                prototype_kwargs = dict(
                    feat_dim=hidden_dim,
                    hidden_dim=getattr(args, 'prr_hidden_dim', 128),
                    embedding_dim=getattr(args, 'proto_embedding_dim', 64),
                    num_prototypes=getattr(args, 'proto_num_fg', 4),
                    warmup_epochs=getattr(args, 'prr_warmup_epochs', 2),
                    prototype_temperature=getattr(
                        args, 'proto_temperature', 0.1
                    ),
                    target_margin=getattr(args, 'prr_target_margin', 0.2),
                    margin_temperature=getattr(
                        args, 'prr_margin_temperature', 0.2
                    ),
                    rescue_weight=getattr(args, 'prr_rescue_weight', 0.005),
                    logit_gradient_scale=getattr(
                        args, 'prr_logit_gradient_scale', 0.005
                    ),
                    prototype_rank_weight=getattr(
                        args, 'prr_proto_rank_weight', 0.005
                    ),
                    metric_weight=getattr(args, 'prr_metric_weight', 0.005),
                    center_weight=getattr(args, 'prr_center_weight', 0.005),
                    balance_weight=getattr(args, 'prr_balance_weight', 0.001),
                    diversity_weight=getattr(
                        args, 'prr_diversity_weight', 0.001
                    ),
                    diversity_margin=getattr(
                        args, 'potp_diversity_margin', 0.5
                    ),
                    positive_radius=getattr(args, 'proto_positive_radius', 15.0),
                    background_radius=getattr(
                        args, 'proto_background_radius', 30.0
                    ),
                    maximum_cell_probability=getattr(
                        args, 'prr_max_cell_probability', 0.55
                    ),
                    support_low_quantile=getattr(
                        args, 'prr_support_low_quantile', 0.2
                    ),
                    support_high_quantile=getattr(
                        args, 'prr_support_high_quantile', 0.8
                    ),
                    max_rescue_per_image=getattr(
                        args, 'prr_max_rescue_per_image', 32
                    ),
                    max_random_background_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 16
                    ),
                    support_reservoir_size=getattr(
                        args, 'proto_gt_support_queue_size', 8192
                    ),
                    kmeans_iterations=getattr(
                        args, 'proto_kmeans_iterations', 20
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                if self.prototype_mode == 'prototype_local_soft_positive':
                    prototype_kwargs.update(
                        local_radius=getattr(args, 'plsp_local_radius', 15.0),
                        minimum_distance_improvement=getattr(
                            args, 'plsp_min_distance_improvement', 3.0
                        ),
                        minimum_similarity_improvement=getattr(
                            args, 'plsp_min_similarity_improvement', 0.1
                        ),
                        maximum_matched_probability=getattr(
                            args, 'plsp_max_matched_probability', 0.56
                        ),
                        max_soft_positive_per_image=getattr(
                            args, 'plsp_max_per_image', 8
                        ),
                        soft_positive_weight=getattr(
                            args, 'plsp_positive_weight', 0.1
                        ),
                    )
                self.prototype_head = prototype_class(**prototype_kwargs)
                implementation_version = (
                    PROTOTYPE_LOCAL_SOFT_POSITIVE_VERSION
                    if self.prototype_mode == 'prototype_local_soft_positive'
                    else PROTOTYPE_RELIABILITY_RESCUE_VERSION
                )
                self.prototype_metadata = {
                    'mode': self.prototype_mode,
                    'implementation_version': implementation_version,
                    'support_source': 'exact_gt_shared_six_level_cls_path',
                    'rescue_source': (
                        'unmatched_nearest_owner_local_better_geometry_and_prototype'
                        if self.prototype_mode == 'prototype_local_soft_positive'
                        else 'matched_within_15px_low_score_proto_reliable'
                    ),
                    'negative_source': 'projector_only_unmatched_beyond_30px',
                    'prototype_state': 'epoch_raw_gt_reprojection_kmeans_quantiles',
                    'detector_background_auxiliary_gradient': False,
                    'matcher_modified': False,
                    'regression_modified': False,
                    'inference_fusion': False,
                }
            elif self.prototype_mode == 'prototype_guided_ranking':
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is disabled for '
                        'prototype_guided_ranking'
                    )
                self.prototype_head = PrototypeGuidedRanking(
                    feat_dim=hidden_dim,
                    hidden_dim=getattr(args, 'pgrp_hidden_dim', 128),
                    embedding_dim=getattr(args, 'proto_embedding_dim', 64),
                    num_prototypes=getattr(args, 'proto_num_fg', 4),
                    warmup_epochs=getattr(args, 'pgrp_warmup_epochs', 2),
                    prototype_temperature=getattr(
                        args, 'proto_temperature', 0.1
                    ),
                    ranking_temperature=getattr(
                        args, 'pgrp_ranking_temperature', 0.1
                    ),
                    prototype_margin=getattr(args, 'pgrp_proto_margin', 0.1),
                    classification_margin=getattr(args, 'pgrp_cls_margin', 0.2),
                    prototype_rank_weight=getattr(
                        args, 'pgrp_proto_rank_weight', 0.005
                    ),
                    classification_rank_weight=getattr(
                        args, 'pgrp_cls_rank_weight', 0.01
                    ),
                    metric_weight=getattr(args, 'pgrp_metric_weight', 0.005),
                    center_weight=getattr(args, 'pgrp_center_weight', 0.005),
                    balance_weight=getattr(args, 'pgrp_balance_weight', 0.001),
                    diversity_weight=getattr(
                        args, 'pgrp_diversity_weight', 0.001
                    ),
                    diversity_margin=getattr(
                        args, 'potp_diversity_margin', 0.5
                    ),
                    input_gradient_scale=getattr(
                        args, 'proto_input_grad_scale', 0.05
                    ),
                    positive_radius=getattr(args, 'proto_positive_radius', 15.0),
                    background_radius=getattr(
                        args, 'proto_background_radius', 30.0
                    ),
                    hard_positive_fraction=getattr(
                        args, 'pgrp_hard_positive_fraction', 0.25
                    ),
                    max_pairs_per_image=getattr(
                        args, 'pgrp_max_pairs_per_image', 32
                    ),
                    max_random_background_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 16
                    ),
                    support_reservoir_size=getattr(
                        args, 'proto_gt_support_queue_size', 8192
                    ),
                    kmeans_iterations=getattr(
                        args, 'proto_kmeans_iterations', 20
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                self.prototype_metadata = {
                    'mode': self.prototype_mode,
                    'implementation_version': PROTOTYPE_GUIDED_RANKING_VERSION,
                    'support_source': 'exact_gt_shared_six_level_cls_path',
                    'positive_source': 'hungarian_match_within_15px',
                    'negative_source': 'unmatched_beyond_30px',
                    'prototype_state': 'epoch_raw_gt_reprojection_kmeans',
                    'inference_fusion': False,
                }
            elif self.prototype_mode == 'positive_only_train_proto':
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is disabled for '
                        'positive_only_train_proto'
                    )
                self.prototype_head = PositiveOnlyTrainPrototype(
                    feat_dim=hidden_dim,
                    hidden_dim=getattr(args, 'potp_hidden_dim', 128),
                    embedding_dim=getattr(args, 'proto_embedding_dim', 64),
                    num_prototypes=getattr(args, 'proto_num_fg', 4),
                    warmup_epochs=getattr(args, 'potp_warmup_epochs', 2),
                    temperature=getattr(args, 'proto_temperature', 0.1),
                    positive_margin=getattr(args, 'potp_positive_margin', 0.4),
                    negative_margin=getattr(args, 'potp_negative_margin', 0.2),
                    diversity_margin=getattr(args, 'potp_diversity_margin', 0.5),
                    pair_weight=getattr(args, 'potp_pair_weight', 0.01),
                    positive_weight=getattr(args, 'potp_positive_weight', 0.01),
                    negative_weight=getattr(args, 'potp_negative_weight', 0.005),
                    balance_weight=getattr(args, 'potp_balance_weight', 0.001),
                    diversity_weight=getattr(args, 'potp_diversity_weight', 0.001),
                    input_gradient_scale=getattr(
                        args, 'proto_input_grad_scale', 0.05
                    ),
                    background_radius=getattr(
                        args, 'proto_background_radius', 30.0
                    ),
                    max_hard_background_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 32
                    ),
                    max_random_background_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 16
                    ),
                    support_reservoir_size=getattr(
                        args, 'proto_gt_support_queue_size', 8192
                    ),
                    kmeans_iterations=getattr(
                        args, 'proto_kmeans_iterations', 20
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                    hard_positive_fraction=getattr(
                        args, 'potp_hard_positive_fraction', 1.0
                    ),
                    max_hard_positive_per_image=getattr(
                        args, 'potp_max_hard_positive_per_image', 0
                    ),
                    detach_support_input=bool(getattr(
                        args, 'potp_detach_support_input', 0
                    )),
                )
                self.prototype_metadata = {
                    'mode': self.prototype_mode,
                    'implementation_version': POSITIVE_ONLY_TRAIN_PROTO_VERSION,
                    'support_source': 'exact_gt_shared_six_level_cls_path',
                    'negative_source': 'unmatched_beyond_gt_radius',
                    'has_background_prototypes': False,
                    'inference_fusion': False,
                }
            elif self.prototype_mode == 'source_supervised_candidate_proto':
                bank_path = str(getattr(args, 'proto_bank_path', ''))
                if not bank_path:
                    raise ValueError(
                        '--proto_bank_path is required for '
                        'source_supervised_candidate_proto'
                    )
                self.prototype_head = SourceSupervisedCandidatePrototype(
                    feat_dim=hidden_dim,
                    num_foreground_prototypes=getattr(args, 'proto_num_fg', 4),
                    num_background_prototypes=getattr(args, 'proto_num_bg', 8),
                    foreground_queue_size=getattr(
                        args, 'proto_fg_queue_size', 8192
                    ),
                    background_queue_size=getattr(
                        args, 'proto_bg_queue_size', 16384
                    ),
                    max_hard_positive_per_image=getattr(
                        args, 'sscp_max_hard_positive_per_image', 32
                    ),
                    max_random_positive_per_image=getattr(
                        args, 'sscp_max_random_positive_per_image', 32
                    ),
                    max_hard_background_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 32
                    ),
                    max_random_background_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 32
                    ),
                    positive_radius=getattr(args, 'proto_positive_radius', 15.0),
                    background_radius=getattr(
                        args, 'proto_background_radius', 30.0
                    ),
                    prototype_temperature=getattr(
                        args, 'proto_temperature', 0.1
                    ),
                    prototype_margin=getattr(args, 'sscp_margin', 0.1),
                    prototype_momentum=getattr(args, 'proto_momentum', 0.95),
                    sinkhorn_epsilon=getattr(
                        args, 'sscp_sinkhorn_epsilon', 0.05
                    ),
                    sinkhorn_iterations=getattr(
                        args, 'sscp_sinkhorn_iterations', 3
                    ),
                    minimum_assignment_share=getattr(
                        args, 'sscp_min_assignment_share', 0.02
                    ),
                    dead_prototype_patience=getattr(
                        args, 'proto_dead_patience', 2
                    ),
                    alpha_max=getattr(args, 'sscp_alpha_max', 0.25),
                    alpha_initial=getattr(args, 'sscp_alpha_initial', 0.001),
                    confidence_threshold=getattr(
                        args, 'sscp_confidence_threshold', 0.05
                    ),
                    fusion_clip=getattr(args, 'sscp_fusion_clip', 0.5),
                    proto_ce_weight=getattr(
                        args, 'sscp_proto_ce_weight', 0.02
                    ),
                    margin_weight=getattr(args, 'sscp_margin_weight', 0.02),
                    fused_ce_weight=getattr(
                        args, 'sscp_fused_ce_weight', 0.05
                    ),
                    input_gradient_scale=getattr(
                        args, 'proto_input_grad_scale', 0.05
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                loaded_metadata = self.prototype_head.load_bank_file(bank_path)
                self.prototype_metadata = {
                    'mode': self.prototype_mode,
                    'implementation_version': (
                        SOURCE_SUPERVISED_CANDIDATE_PROTO_VERSION
                    ),
                    'support_source': (
                        'raw_match_reliable_positive_and_verified_far_background'
                    ),
                    'native_feature_dimension': hidden_dim,
                    'inference_fusion': self.prototype_inference_fusion,
                    **loaded_metadata,
                }
            elif self.prototype_mode == 'discriminative_dual_proxy':
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is disabled for discriminative_dual_proxy'
                    )
                self.prototype_head = DiscriminativeDualProxy(
                    feat_dim=hidden_dim,
                    embedding_dim=getattr(args, 'proto_embedding_dim', 64),
                    num_fg=getattr(args, 'proto_dual_num_fg', 4),
                    num_bg=getattr(args, 'proto_dual_num_bg', 4),
                    warmup_epochs=getattr(args, 'proto_gt_warmup_epochs', 5),
                    foreground_queue_size=getattr(
                        args, 'proto_dual_fg_queue_size', 8192
                    ),
                    background_queue_size=getattr(
                        args, 'proto_dual_bg_queue_size', 8192
                    ),
                    max_foreground_per_image=getattr(
                        args, 'proto_gt_max_support_per_image', 64
                    ),
                    max_hard_background_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 32
                    ),
                    max_random_background_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 32
                    ),
                    prototype_momentum=getattr(args, 'proto_momentum', 0.99),
                    projector_momentum=getattr(
                        args, 'proto_gt_projector_momentum', 0.999
                    ),
                    positive_radius=getattr(args, 'proto_positive_radius', 15.0),
                    background_radius=getattr(
                        args, 'proto_background_radius', 30.0
                    ),
                    temperature=getattr(args, 'proto_temperature', 0.2),
                    supcon_weight=getattr(args, 'proto_dual_supcon_weight', 0.5),
                    separation_weight=getattr(
                        args, 'proto_dual_separation_weight', 0.5
                    ),
                    separation_margin=getattr(
                        args, 'proto_dual_separation_margin', 0.1
                    ),
                    balance_weight=getattr(
                        args, 'proto_dual_balance_weight', 0.05
                    ),
                    minimum_assignment_share=getattr(
                        args, 'proto_gt_min_assignment_share', 0.05
                    ),
                    dead_prototype_patience=getattr(
                        args, 'proto_dead_patience', 2
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                self.prototype_metadata = {
                    'mode': self.prototype_mode,
                    'implementation_version': DISCRIMINATIVE_DUAL_PROXY_VERSION,
                    'support_source': 'exact_gt_and_verified_far_background',
                    'inference_fusion': False,
                }
                bank_path = str(getattr(args, 'proto_bank_path', ''))
                if bank_path:
                    loaded_metadata = self.prototype_head.load_bank_file(bank_path)
                    self.prototype_metadata.update(loaded_metadata)
            elif self.prototype_mode == 'gt_foreground_proxy':
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is disabled for gt_foreground_proxy'
                    )
                self.prototype_head = GTForegroundProxy(
                    feat_dim=hidden_dim,
                    embedding_dim=getattr(args, 'proto_embedding_dim', 64),
                    num_prototypes=getattr(args, 'proto_num_fg', 4),
                    warmup_epochs=getattr(args, 'proto_gt_warmup_epochs', 5),
                    support_queue_size=getattr(
                        args, 'proto_gt_support_queue_size', 8192
                    ),
                    max_support_per_image=getattr(
                        args, 'proto_gt_max_support_per_image', 64
                    ),
                    prototype_momentum=getattr(args, 'proto_momentum', 0.99),
                    projector_momentum=getattr(
                        args, 'proto_gt_projector_momentum', 0.999
                    ),
                    query_radius=getattr(args, 'proto_positive_radius', 15.0),
                    background_radius=getattr(
                        args, 'proto_background_radius', 30.0
                    ),
                    max_hard_background_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 16
                    ),
                    background_margin=getattr(
                        args, 'proto_gt_background_margin', 0.2
                    ),
                    minimum_assignment_share=getattr(
                        args, 'proto_gt_min_assignment_share', 0.05
                    ),
                    dead_prototype_patience=getattr(
                        args, 'proto_dead_patience', 2
                    ),
                    temperature=getattr(args, 'proto_temperature', 0.2),
                    align_weight=getattr(args, 'proto_gt_align_weight', 1.0),
                    support_weight=getattr(args, 'proto_gt_support_weight', 1.0),
                    query_weight=getattr(args, 'proto_gt_query_weight', 1.0),
                    background_weight=getattr(
                        args, 'proto_gt_background_weight', 0.5
                    ),
                    balance_weight=getattr(
                        args, 'proto_gt_balance_weight', 0.05
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                self.prototype_metadata = {
                    'mode': self.prototype_mode,
                    'implementation_version': GT_FOREGROUND_PROXY_VERSION,
                    'support_source': 'exact_gt_points_shared_six_level_cls_path',
                    'inference_fusion': False,
                }
            elif self.prototype_mode == 'frozen_supervised_metric':
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is disabled for '
                        'frozen_supervised_metric'
                    )
                bank_path = str(getattr(args, 'proto_bank_path', ''))
                if not bank_path:
                    raise ValueError(
                        '--proto_bank_path is required for frozen_supervised_metric'
                    )
                self.prototype_head = FrozenSupervisedMetricPrototype(
                    feat_dim=hidden_dim,
                    embedding_dim=getattr(args, 'proto_embedding_dim', 32),
                    prototype_counts=(
                        getattr(args, 'proto_num_fg', 4),
                        getattr(args, 'proto_num_hard_bg', 4),
                        getattr(args, 'proto_num_random_bg', 2),
                    ),
                    temperature=getattr(args, 'proto_temperature', 0.15),
                    positive_radius=getattr(
                        args, 'proto_initial_positive_radius', 10.0
                    ),
                    background_radius=getattr(args, 'proto_background_radius', 30.0),
                    max_positive_per_image=getattr(
                        args, 'proto_max_pos_per_image', 32
                    ),
                    max_hard_negative_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 16
                    ),
                    max_random_negative_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 16
                    ),
                    hard_negative_weight=getattr(
                        args, 'proto_hard_bg_term_weight', 2.0
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                self.prototype_metadata = self.prototype_head.load_bank_file(
                    bank_path
                )
            elif self.prototype_mode == 'frozen_teacher_fg':
                if self.prototype_inference_fusion:
                    raise ValueError(
                        'prototype inference fusion is forbidden for frozen_teacher_fg'
                    )
                bank_path = str(getattr(args, 'proto_bank_path', ''))
                if not bank_path:
                    raise ValueError(
                        '--proto_bank_path is required for frozen_teacher_fg'
                    )
                self.prototype_head = FrozenTeacherForegroundPrototype(
                    feat_dim=hidden_dim,
                    num_prototypes=getattr(args, 'proto_num_fg', 4),
                    negative_bank_size=getattr(args, 'proto_bg_queue_size', 8192),
                    negative_sample_size=getattr(
                        args, 'proto_fixed_negative_sample_size', 256
                    ),
                    temperature=getattr(args, 'proto_temperature', 0.1),
                    positive_radius=getattr(
                        args, 'proto_initial_positive_radius', 10.0
                    ),
                    background_radius=getattr(args, 'proto_background_radius', 30.0),
                    max_positive_per_image=getattr(
                        args, 'proto_max_pos_per_image', 64
                    ),
                    max_hard_negative_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 16
                    ),
                    max_random_negative_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 16
                    ),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                )
                self.prototype_metadata = self.prototype_head.load_fixed_bank_file(
                    bank_path
                )
            else:
                self.prototype_head = CandidatePrototypeBank(
                    feat_dim=hidden_dim,
                    embedding_dim=getattr(args, 'proto_embedding_dim', 128),
                    num_fg_prototypes=getattr(args, 'proto_num_fg', 4),
                    num_bg_prototypes=getattr(args, 'proto_num_bg', 4),
                    foreground_queue_size=getattr(args, 'proto_fg_queue_size', 4096),
                    background_queue_size=getattr(args, 'proto_bg_queue_size', 8192),
                    temperature=getattr(args, 'proto_temperature', 0.1),
                    fusion_alpha=getattr(args, 'proto_fusion_alpha', 0.1),
                    fusion_clip=getattr(args, 'proto_fusion_clip', 2.0),
                    positive_radius=getattr(args, 'proto_positive_radius', 15.0),
                    initial_positive_radius=getattr(
                        args, 'proto_initial_positive_radius', 10.0
                    ),
                    background_radius=getattr(args, 'proto_background_radius', 30.0),
                    max_positive_per_image=getattr(args, 'proto_max_pos_per_image', 64),
                    max_background_per_image=getattr(args, 'proto_max_bg_per_image', 32),
                    max_hard_background_per_image=getattr(
                        args, 'proto_max_hard_bg_per_image', 16
                    ),
                    max_random_background_per_image=getattr(
                        args, 'proto_max_random_bg_per_image', 16
                    ),
                    kmeans_iterations=getattr(args, 'proto_kmeans_iterations', 10),
                    prototype_momentum=getattr(args, 'proto_momentum', 0.99),
                    dead_prototype_patience=getattr(args, 'proto_dead_patience', 3),
                    input_gradient_scale=getattr(args, 'proto_input_grad_scale', 0.1),
                    sampling_seed=getattr(args, 'proto_sampling_seed', 0),
                    loss_mode=getattr(args, 'proto_loss_mode', 'pooled'),
                    positive_term_weight=getattr(
                        args, 'proto_positive_term_weight', 1.0
                    ),
                    hard_background_term_weight=getattr(
                        args, 'proto_hard_bg_term_weight', 2.0
                    ),
                    random_background_term_weight=getattr(
                        args, 'proto_random_bg_term_weight', 0.25
                    ),
                    update_mode=getattr(args, 'proto_update_mode', 'assignment_ema'),
                    minimum_assignment_share=getattr(
                        args, 'proto_min_assignment_share', 0.0
                    ),
                    refresh_interval_steps=getattr(
                        args, 'proto_refresh_interval_steps', 0
                    ),
                )

    @staticmethod
    def _pack_support_points(prototype_support_points, device):
        if prototype_support_points is None:
            return None, None
        batch_size = len(prototype_support_points)
        max_points = max(
            (int(points.shape[0]) for points in prototype_support_points),
            default=0,
        )
        support_points = torch.zeros(
            batch_size, max_points, 2, dtype=torch.float32, device=device
        )
        support_valid_mask = torch.zeros(
            batch_size, max_points, dtype=torch.bool, device=device
        )
        for batch_index, points in enumerate(prototype_support_points):
            count = int(points.shape[0])
            if count:
                support_points[batch_index, :count] = points.to(
                    device=device, dtype=torch.float32
                )
                support_valid_mask[batch_index, :count] = True
        return support_points, support_valid_mask

    def forward(self, images, prototype_support_points=None):
        self._prototype_debug_tensors = None
        anchors = self.get_aps(images)
        features = self.backbone(images)

        reg_features, cls_features, reg_attn, cls_attn = self.extract_features(features, anchors)
        pnt_coords = self.reg_head(reg_features) + anchors
        raw_cls_logits = self.cls_head(cls_features)
        cls_logits = raw_cls_logits

        prototype_outputs = None
        support_valid_mask = None
        run_prototype_head = self.prototype_head is not None and not (
            self.prototype_mode in {
                'prototype_reliability_rescue',
                'prototype_local_soft_positive',
            }
            and not self.training
            and not self.prototype_diagnostic_eval
        )
        if run_prototype_head:
            if self.prototype_mode == 'source_supervised_candidate_proto':
                prototype_outputs = self.prototype_head(
                    cls_features,
                    raw_cls_logits,
                )
            elif self.prototype_mode in {
                'gt_foreground_proxy', 'discriminative_dual_proxy',
                'positive_only_train_proto', 'prototype_guided_ranking',
                'prototype_reliability_rescue', 'prototype_local_soft_positive',
            }:
                support_points, support_valid_mask = self._pack_support_points(
                    prototype_support_points, images.device
                )
                support_features = None
                if support_points is not None and support_points.shape[1] > 0:
                    _, support_features, _, _ = self.extract_features(features, support_points)
                prototype_outputs = self.prototype_head(
                    cls_features,
                    raw_cls_logits,
                    support_features=support_features,
                )
            else:
                prototype_outputs = self.prototype_head(
                    cls_features,
                    raw_cls_logits,
                    apply_fusion=(not self.training and self.prototype_inference_fusion),
                )
            if self.prototype_mode == 'source_supervised_candidate_proto':
                # Raw predictions own train-time matching and the original P2P loss.
                # Prototype fusion is consumed by the auxiliary loss during training
                # and becomes visible to the detector only during fused inference.
                if not self.training and self.prototype_inference_fusion:
                    cls_logits = prototype_outputs['fused_logits']
            else:
                cls_logits = prototype_outputs['fused_logits']

        # outputs = {'pnt_coords': pnt_coords, 'cls_logits': cls_logits}
        outputs = {
            'pnt_coords': pnt_coords,
            'anchor_points': anchors,
            'cls_logits': cls_logits,
            'raw_cls_logits': raw_cls_logits,
            'reg_attn': reg_attn,
            'cls_attn': cls_attn,
        }
        if prototype_outputs is not None:
            outputs.update({
                'proto_embeddings': prototype_outputs['embeddings'],
                'proto_logits': prototype_outputs['prototype_logits'],
            })
            if 'prototype_distances' in prototype_outputs:
                outputs['proto_distances'] = prototype_outputs[
                    'prototype_distances'
                ]
            if self.prototype_mode == 'source_supervised_candidate_proto':
                outputs['proto_fused_logits'] = prototype_outputs['fused_logits']
                outputs['proto_margin'] = prototype_outputs['prototype_margin']
                outputs['proto_fusion_delta'] = prototype_outputs['fusion_delta']
            if self.prototype_mode == 'discriminative_dual_proxy':
                outputs['proto_momentum_embeddings'] = prototype_outputs[
                    'momentum_embeddings'
                ]
            if self.prototype_mode == 'gt_foreground_proxy':
                outputs['proto_support_embeddings'] = prototype_outputs[
                    'support_embeddings'
                ]
                outputs['proto_momentum_support_embeddings'] = prototype_outputs[
                    'momentum_support_embeddings'
                ]
                if support_valid_mask is None:
                    support_valid_mask = torch.zeros(
                        images.shape[0], 0, dtype=torch.bool, device=images.device
                    )
                outputs['proto_support_valid_mask'] = support_valid_mask
            elif self.prototype_mode in {
                'positive_only_train_proto', 'prototype_guided_ranking',
                'prototype_reliability_rescue', 'prototype_local_soft_positive',
            }:
                outputs['proto_support_embeddings'] = prototype_outputs[
                    'support_embeddings'
                ]
                if self.prototype_mode in {
                    'prototype_guided_ranking', 'prototype_reliability_rescue',
                    'prototype_local_soft_positive',
                }:
                    outputs['proto_candidate_raw_features'] = prototype_outputs[
                        'candidate_raw_features'
                    ]
                    outputs['proto_support_raw_features'] = prototype_outputs[
                        'support_raw_features'
                    ]
                if support_valid_mask is None:
                    support_valid_mask = torch.zeros(
                        images.shape[0], 0, dtype=torch.bool, device=images.device
                    )
                outputs['proto_support_valid_mask'] = support_valid_mask
            elif self.prototype_mode == 'discriminative_dual_proxy':
                outputs['proto_support_embeddings'] = prototype_outputs[
                    'support_embeddings'
                ]
                outputs['proto_momentum_support_embeddings'] = prototype_outputs[
                    'momentum_support_embeddings'
                ]
                if support_valid_mask is None:
                    support_valid_mask = torch.zeros(
                        images.shape[0], 0, dtype=torch.bool, device=images.device
                    )
                outputs['proto_support_valid_mask'] = support_valid_mask
            if self.training and getattr(
                self.prototype_head, 'refresh_interval_steps', 0
            ) > 0:
                outputs['proto_source_features'] = cls_features
            if self.training and self.prototype_debug_enabled:
                # Keep the tensors created inside the wrapped module. DDP wraps
                # returned tensors independently, which can make cross-output
                # autograd diagnostics report an unavailable gradient.
                self._prototype_debug_tensors = {
                    'cls_features': cls_features,
                    'fpn_features': tuple(features),
                }
                outputs['proto_cls_features'] = cls_features
                outputs['proto_fpn_features'] = tuple(features)
        return outputs

    def get_prototype_debug_tensors(self):
        return self._prototype_debug_tensors

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
        args=args,
    )

    return model
