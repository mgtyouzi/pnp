import os
import sys
import argparse
import csv
import hashlib
import json
import math
import random
#import wandb
import cv2 as cv
import numpy as np
import time

from utils import *
from tqdm import tqdm

# from dataset import build_dataset
from dataset_zy_src import build_dataset
from models.detr import build_model
from loss import build_criterion
from matcher import build_matcher
from lr_sched import adjust_learning_rate
from prototype_gradient_schedule import select_gradient_probe
from checkpoint_schedule import checkpoint_label, should_save_checkpoint
from diagnostic_aggregation import accumulate_finite_diagnostics

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29500'
# os.environ['RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Train
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_eval', default=100, type=int)
    parser.add_argument(
        '--checkpoint_interval', default=0, type=int,
        help='save a numbered checkpoint every N completed epochs; 0 keeps legacy saving only',
    )

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--init_checkpoint', default='',
                        help='load model weights only and start a new optimizer/run')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_classes', type=int, default=1,
                        help="Number of cell categories")

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.3, type=float,
                        help="Relative classification weight of the no-object class")

    # * Matcher
    parser.add_argument('--set_cost_point', default=0.1, type=float,
                        help="L2 point coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    # * Model
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")

    # Candidate-level foreground/background prototypes.
    parser.add_argument('--proto_enable', action='store_true')
    parser.add_argument(
        '--proto_mode', default='legacy_online',
        choices=(
            'legacy_online', 'frozen_teacher_fg', 'frozen_supervised_metric',
            'gt_foreground_proxy', 'discriminative_dual_proxy',
            'source_supervised_candidate_proto',
            'positive_only_train_proto',
            'prototype_guided_ranking',
            'prototype_reliability_rescue',
            'prototype_local_soft_positive',
        )
    )
    parser.add_argument('--proto_bank_path', default='', type=str)
    parser.add_argument(
        '--proto_fixed_negative_sample_size', default=256, type=int
    )
    parser.add_argument('--proto_start_epoch', default=20, type=int)
    parser.add_argument('--proto_embedding_dim', default=128, type=int)
    parser.add_argument('--proto_num_fg', default=4, type=int)
    parser.add_argument('--proto_num_bg', default=8, type=int)
    parser.add_argument('--proto_num_hard_bg', default=4, type=int)
    parser.add_argument('--proto_num_random_bg', default=2, type=int)
    parser.add_argument('--proto_fg_queue_size', default=4096, type=int)
    parser.add_argument('--proto_bg_queue_size', default=8192, type=int)
    parser.add_argument('--proto_temperature', default=0.2, type=float)
    parser.add_argument('--proto_loss_weight', default=0.01, type=float)
    parser.add_argument('--proto_initial_positive_radius', default=10.0, type=float)
    parser.add_argument('--proto_positive_radius', default=15.0, type=float)
    parser.add_argument('--proto_background_radius', default=30.0, type=float)
    parser.add_argument('--proto_max_pos_per_image', default=64, type=int)
    parser.add_argument('--proto_max_bg_per_image', default=32, type=int)
    parser.add_argument('--proto_max_hard_bg_per_image', default=16, type=int)
    parser.add_argument('--proto_max_random_bg_per_image', default=16, type=int)
    parser.add_argument('--proto_kmeans_iterations', default=10, type=int)
    parser.add_argument('--proto_momentum', default=0.99, type=float)
    parser.add_argument('--proto_dead_patience', default=3, type=int)
    parser.add_argument('--proto_input_grad_scale', default=0.1, type=float)
    parser.add_argument('--proto_sampling_seed', default=0, type=int)
    parser.add_argument(
        '--proto_loss_mode', default='pooled',
        choices=('pooled', 'source_weighted')
    )
    parser.add_argument('--proto_positive_term_weight', default=1.0, type=float)
    parser.add_argument('--proto_hard_bg_term_weight', default=2.0, type=float)
    parser.add_argument('--proto_random_bg_term_weight', default=0.25, type=float)
    parser.add_argument(
        '--proto_update_mode', default='assignment_ema',
        choices=('assignment_ema', 'kmeans_ema')
    )
    parser.add_argument('--proto_min_assignment_share', default=0.0, type=float)
    parser.add_argument(
        '--proto_refresh_interval_steps',
        default=0,
        type=int,
        help='refresh prototypes from a recent support window every N optimizer steps; 0 keeps epoch updates',
    )
    parser.add_argument('--proto_debug_interval', default=200, type=int)
    parser.add_argument('--proto_debug_fail_fast', default=1, type=int, choices=(0, 1))
    parser.add_argument('--proto_inference_fusion', default=0, type=int, choices=(0, 1))
    parser.add_argument('--proto_fusion_alpha', default=0.1, type=float)
    parser.add_argument('--proto_fusion_clip', default=2.0, type=float)
    parser.add_argument('--proto_gt_warmup_epochs', default=5, type=int)
    parser.add_argument('--proto_gt_support_queue_size', default=8192, type=int)
    parser.add_argument('--proto_gt_max_support_per_image', default=64, type=int)
    parser.add_argument('--proto_gt_projector_momentum', default=0.999, type=float)
    parser.add_argument('--proto_gt_background_margin', default=0.2, type=float)
    parser.add_argument('--proto_gt_min_assignment_share', default=0.05, type=float)
    parser.add_argument('--proto_gt_align_weight', default=1.0, type=float)
    parser.add_argument('--proto_gt_support_weight', default=1.0, type=float)
    parser.add_argument('--proto_gt_query_weight', default=1.0, type=float)
    parser.add_argument('--proto_gt_background_weight', default=0.5, type=float)
    parser.add_argument('--proto_gt_balance_weight', default=0.05, type=float)
    parser.add_argument('--proto_dual_num_fg', default=4, type=int)
    parser.add_argument('--proto_dual_num_bg', default=4, type=int)
    parser.add_argument('--proto_dual_fg_queue_size', default=8192, type=int)
    parser.add_argument('--proto_dual_bg_queue_size', default=8192, type=int)
    parser.add_argument('--proto_dual_supcon_weight', default=0.5, type=float)
    parser.add_argument('--proto_dual_separation_weight', default=0.5, type=float)
    parser.add_argument('--proto_dual_separation_margin', default=0.1, type=float)
    parser.add_argument('--proto_dual_balance_weight', default=0.05, type=float)
    parser.add_argument('--proto_freeze_detector', default=0, type=int, choices=(0, 1))
    parser.add_argument('--sscp_max_hard_positive_per_image', default=32, type=int)
    parser.add_argument('--sscp_max_random_positive_per_image', default=32, type=int)
    parser.add_argument('--sscp_proto_ce_weight', default=0.02, type=float)
    parser.add_argument('--sscp_margin_weight', default=0.02, type=float)
    parser.add_argument('--sscp_fused_ce_weight', default=0.05, type=float)
    parser.add_argument('--sscp_margin', default=0.1, type=float)
    parser.add_argument('--sscp_alpha_max', default=0.25, type=float)
    parser.add_argument('--sscp_alpha_initial', default=0.001, type=float)
    parser.add_argument('--sscp_confidence_threshold', default=0.05, type=float)
    parser.add_argument('--sscp_fusion_clip', default=0.5, type=float)
    parser.add_argument('--sscp_min_assignment_share', default=0.02, type=float)
    parser.add_argument('--sscp_sinkhorn_epsilon', default=0.05, type=float)
    parser.add_argument('--sscp_sinkhorn_iterations', default=3, type=int)
    parser.add_argument('--potp_warmup_epochs', default=2, type=int)
    parser.add_argument('--potp_hidden_dim', default=128, type=int)
    parser.add_argument('--potp_positive_margin', default=0.4, type=float)
    parser.add_argument('--potp_negative_margin', default=0.2, type=float)
    parser.add_argument('--potp_diversity_margin', default=0.5, type=float)
    parser.add_argument('--potp_pair_weight', default=0.01, type=float)
    parser.add_argument('--potp_positive_weight', default=0.01, type=float)
    parser.add_argument('--potp_negative_weight', default=0.005, type=float)
    parser.add_argument('--potp_balance_weight', default=0.001, type=float)
    parser.add_argument('--potp_diversity_weight', default=0.001, type=float)
    parser.add_argument('--potp_hard_positive_fraction', default=1.0, type=float)
    parser.add_argument('--potp_max_hard_positive_per_image', default=0, type=int)
    parser.add_argument(
        '--potp_detach_support_input', default=0, type=int, choices=(0, 1)
    )
    parser.add_argument('--pgrp_hidden_dim', default=128, type=int)
    parser.add_argument('--pgrp_warmup_epochs', default=2, type=int)
    parser.add_argument('--pgrp_ranking_temperature', default=0.1, type=float)
    parser.add_argument('--pgrp_proto_margin', default=0.1, type=float)
    parser.add_argument('--pgrp_cls_margin', default=0.2, type=float)
    parser.add_argument('--pgrp_proto_rank_weight', default=0.005, type=float)
    parser.add_argument('--pgrp_cls_rank_weight', default=0.01, type=float)
    parser.add_argument('--pgrp_metric_weight', default=0.005, type=float)
    parser.add_argument('--pgrp_center_weight', default=0.005, type=float)
    parser.add_argument('--pgrp_balance_weight', default=0.001, type=float)
    parser.add_argument('--pgrp_diversity_weight', default=0.001, type=float)
    parser.add_argument('--pgrp_hard_positive_fraction', default=0.25, type=float)
    parser.add_argument('--pgrp_max_pairs_per_image', default=32, type=int)
    parser.add_argument('--prr_hidden_dim', default=128, type=int)
    parser.add_argument('--prr_warmup_epochs', default=2, type=int)
    parser.add_argument('--prr_support_low_quantile', default=0.2, type=float)
    parser.add_argument('--prr_support_high_quantile', default=0.8, type=float)
    parser.add_argument('--prr_max_cell_probability', default=0.55, type=float)
    parser.add_argument('--prr_target_margin', default=0.2, type=float)
    parser.add_argument('--prr_margin_temperature', default=0.2, type=float)
    parser.add_argument('--prr_rescue_weight', default=0.005, type=float)
    parser.add_argument('--prr_logit_gradient_scale', default=0.005, type=float)
    parser.add_argument('--prr_proto_rank_weight', default=0.005, type=float)
    parser.add_argument('--prr_metric_weight', default=0.005, type=float)
    parser.add_argument('--prr_center_weight', default=0.005, type=float)
    parser.add_argument('--prr_balance_weight', default=0.001, type=float)
    parser.add_argument('--prr_diversity_weight', default=0.001, type=float)
    parser.add_argument('--prr_max_rescue_per_image', default=32, type=int)
    parser.add_argument('--plsp_local_radius', default=15.0, type=float)
    parser.add_argument(
        '--plsp_min_distance_improvement', default=3.0, type=float
    )
    parser.add_argument(
        '--plsp_min_similarity_improvement', default=0.1, type=float
    )
    parser.add_argument('--plsp_max_matched_probability', default=0.56, type=float)
    parser.add_argument('--plsp_max_per_image', default=8, type=int)
    parser.add_argument('--plsp_positive_weight', default=0.1, type=float)
    parser.add_argument('--reset_rng_after_init', default=0, type=int, choices=(0, 1),
                        help='reset Python/NumPy/Torch RNG after model/checkpoint init for paired attribution')
    parser.add_argument('--deterministic_training', default=0, type=int, choices=(0, 1),
                        help='disable cuDNN benchmark and request deterministic cuDNN kernels')

    # * Dataset
    parser.add_argument('--dataset', default='', type=str)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument(
        '--debug_max_train_batches',
        default=0,
        type=int,
        help='debug only: stop each training epoch after this many batches; 0 uses all batches',
    )
    parser.add_argument(
        '--debug_save_final_checkpoint',
        default=0,
        type=int,
        choices=(0, 1),
        help='debug only: save recent_model.pth after the final training epoch without evaluation',
    )

    # * Evaluator
    parser.add_argument('--match_dis', default=15, type=int)

    # * Distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_teacher_initialization(args, model):
    if not bool(getattr(args, 'proto_enable', False)) or str(
        getattr(args, 'proto_mode', 'legacy_online')
    ) not in {
        'frozen_teacher_fg',
        'frozen_supervised_metric',
        'source_supervised_candidate_proto',
    }:
        return
    metadata = getattr(model, 'prototype_metadata', {})
    teacher_hash = str(metadata.get('checkpoint_sha256', ''))
    if not teacher_hash:
        raise RuntimeError('prototype bank is missing its initialization checkpoint hash')
    if args.resume:
        return
    if not args.init_checkpoint:
        raise ValueError(
            'this prototype mode requires --init_checkpoint from the same source '
            'feature space; training from random weights is forbidden'
        )
    initialization_hash = _sha256_file(args.init_checkpoint)
    if initialization_hash != teacher_hash:
        raise RuntimeError(
            'student initialization does not match the teacher checkpoint hash: '
            f'bank={teacher_hash}, init={initialization_hash}'
        )


def _should_protect_fixed_prototype_bank(args):
    return bool(getattr(args, 'proto_enable', False)) and str(
        getattr(args, 'proto_mode', 'legacy_online')
    ) in {'frozen_teacher_fg', 'frozen_supervised_metric'}


def _freeze_detector_for_proxy_gate(model, enabled):
    if not bool(enabled):
        model.optimizer_projector_only = False
        return []
    if str(getattr(model, 'prototype_mode', '')) != 'discriminative_dual_proxy':
        raise ValueError(
            '--proto_freeze_detector requires --proto_mode=discriminative_dual_proxy'
        )
    trainable = []
    for name, parameter in model.named_parameters():
        keep = name.startswith('prototype_head.projector.')
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(name)
    if not trainable:
        raise RuntimeError('dual-proxy projector has no trainable parameters')
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith('prototype_head.projector.')
    ]
    if unexpected:
        raise RuntimeError(f'non-projector parameters remain trainable: {unexpected}')
    model.optimizer_projector_only = True
    model.prototype_metadata['optimizer_projector_only'] = True
    model.prototype_metadata['trainable_parameter_names'] = trainable
    return trainable


def _apply_projector_only_train_mode(model):
    base_model = model.module if hasattr(model, 'module') else model
    if not bool(getattr(base_model, 'optimizer_projector_only', False)):
        return
    base_model.eval()
    if base_model.prototype_head is None:
        raise RuntimeError('projector-only mode requires a prototype head')
    base_model.prototype_head.projector.train()


def construct_dataset():
    dataset_train = build_dataset(args, 'train')
    dataset_val = build_dataset(args, 'test')

    if args.distributed:
        train_sampler = DistributedSampler(dataset_train)
        data_loader_train = DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size,
                                       num_workers=0, collate_fn=collate_fn_pad)
    else:
        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       num_workers=0, collate_fn=collate_fn_pad)
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=1, num_workers=0,
                                 collate_fn=collate_fn_pad)
    data_loaders = {'train': data_loader_train, 'test': data_loader_val}
    return data_loaders


def _reset_all_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_fingerprint(rank):
    def digest(payload):
        return hashlib.sha256(payload).hexdigest()[:16]

    torch_state = torch.random.get_rng_state().cpu().numpy().tobytes()
    numpy_state = repr(np.random.get_state()).encode("utf-8")
    python_state = repr(random.getstate()).encode("utf-8")
    payload = {
        "python": digest(python_state),
        "numpy": digest(numpy_state),
        "torch_cpu": digest(torch_state),
    }
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(rank).cpu().numpy().tobytes()
        payload["torch_cuda"] = digest(cuda_state)
    return payload


@torch.no_grad()
def _tensor_fingerprint(tensor):
    detached = tensor.detach().cpu().contiguous()
    floating = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
        "sum": float(floating.sum().item()),
        "l2_norm": float(floating.square().sum().sqrt().item()),
        "min": float(floating.min().item()) if floating.numel() else None,
        "max": float(floating.max().item()) if floating.numel() else None,
    }


@torch.no_grad()
def _parameter_group_summary(model, include_hash=False):
    groups = {}
    digests = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        entry = groups.setdefault(
            group, {"parameters": 0, "elements": 0, "sum": 0.0, "square_sum": 0.0}
        )
        if include_hash:
            digest = digests.setdefault(group, hashlib.sha256())
            digest.update(name.encode("utf-8"))
            digest.update(
                parameter.detach().cpu().contiguous().numpy().tobytes()
            )
        value = parameter.detach().float()
        entry["parameters"] += 1
        entry["elements"] += int(value.numel())
        entry["sum"] += float(value.sum().item())
        entry["square_sum"] += float(value.square().sum().item())
    for group, entry in groups.items():
        entry["l2_norm"] = math.sqrt(max(entry.pop("square_sum"), 0.0))
        if include_hash:
            entry["sha256"] = digests[group].hexdigest()
    return groups


def _collect_dataset_debug(dataset, epoch):
    if hasattr(dataset, "get_debug_state"):
        local_state = dataset.get_debug_state(reset=True)
    else:
        local_state = {"phase": getattr(dataset, "phase", "unknown"), "counts": {}, "records": []}
    local_state["epoch"] = int(epoch)
    local_state["rank"] = int(get_rank())
    if is_dist_avail_and_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_state)
        return gathered
    return [local_state]


def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


PROTOTYPE_EPOCH_CSV_FIELDS = [
    "epoch", "regression_loss", "classification_loss", "weighted_prototype_loss",
    "prototype_loss_raw", "prototype_positive", "prototype_rejected_positive",
    "prototype_hard_background", "prototype_random_background",
    "prototype_positive_margin_mean", "prototype_positive_margin_p10",
    "prototype_positive_margin_p50", "prototype_positive_margin_p90",
    "prototype_positive_correct_rate", "prototype_background_margin_mean",
    "prototype_background_margin_p10", "prototype_background_margin_p50",
    "prototype_background_margin_p90", "prototype_background_correct_rate",
    "prototype_hard_background_correct_rate", "prototype_random_background_correct_rate",
    "prototype_loss_positive", "prototype_loss_background",
    "prototype_loss_hard_background", "prototype_loss_random_background",
    "prototype_loss_mode_source_weighted",
    "raw_score_positive_mean", "raw_score_rejected_positive_mean",
    "raw_score_hard_background_mean", "raw_score_random_background_mean",
    "prototype_match_distance_p50", "prototype_match_distance_p90",
    "gradient_all_proto_cls_ratio", "gradient_all_cosine",
    "optimizer_total_grad_norm_before_clip", "optimizer_total_grad_norm_max",
    "optimizer_clip_rate", "train_batches_processed", "train_batches_available",
    "train_batch_fraction",
    "foreground_queue", "background_queue", "foreground_effective_prototypes",
    "background_effective_prototypes", "foreground_background_similarity_mean",
    "foreground_background_similarity_max", "worst_cross_foreground_index",
    "worst_cross_background_index", "worst_cross_foreground_assignment_share",
    "worst_cross_background_assignment_share", "foreground_center_drift_cosine",
    "background_center_drift_cosine", "foreground_fresh_old_alignment_cosine",
    "background_fresh_old_alignment_cosine", "foreground_updated_fresh_cosine",
    "background_updated_fresh_cosine", "fresh_foreground_background_similarity_mean",
    "fresh_foreground_background_similarity_max", "projector_probe_drift_cosine",
    "projector_probe_initialized_now", "foreground_reinitialized",
    "background_reinitialized", "sampling_step", "prototype_refresh_count",
    "prototype_epoch_refresh_events",
    "prototype_epoch_refresh_pre_positive_correct_rate",
    "prototype_epoch_refresh_post_positive_correct_rate",
    "prototype_epoch_refresh_pre_background_correct_rate",
    "prototype_epoch_refresh_post_background_correct_rate",
    "prototype_epoch_refresh_positive_correct_gain",
    "prototype_epoch_refresh_background_correct_gain",
    "prototype_epoch_refresh_projector_drift_min",
    "prototype_epoch_refresh_center_age_max",
    "prototype_epoch_refresh_reprojected_rate",
    "ft_proto_positive", "ft_proto_positive_used",
    "ft_proto_rejected_positive", "ft_proto_hard_negative",
    "ft_proto_random_negative", "ft_proto_ignored_near",
    "ft_proto_loss_raw", "ft_proto_online_negative_used_for_bank",
    "ft_proto_fixed_state_unchanged", "ft_proto_negative_bank_count",
    "ft_proto_negative_sample_count", "ft_proto_positive_similarity_mean",
    "ft_proto_positive_similarity_p10", "ft_proto_positive_similarity_p50",
    "ft_proto_positive_similarity_p90", "ft_proto_negative_similarity_mean",
    "ft_proto_negative_similarity_p10", "ft_proto_negative_similarity_p50",
    "ft_proto_negative_similarity_p90", "ft_proto_raw_score_positive_mean",
    "ft_proto_raw_score_hard_negative_mean",
    "ft_proto_positive_acceptance_rate",
    "ft_proto_positive_vs_fixed_negative_accuracy",
    "ft_proto_effective_foreground_prototypes",
    "ft_proto_similarity_gap_mean", "ft_proto_similarity_gap_p10",
    "ft_proto_similarity_gap_p50", "ft_proto_similarity_gap_p90",
    "ft_proto_online_hard_similarity_mean",
    "ft_proto_online_random_similarity_mean",
    "ft_proto_assignment_share_0", "ft_proto_assignment_share_1",
    "ft_proto_assignment_share_2", "ft_proto_assignment_share_3",
    "gtproxy_loss_raw", "gtproxy_loss_alignment", "gtproxy_loss_support",
    "gtproxy_loss_query", "gtproxy_loss_background", "gtproxy_loss_balance",
    "gtproxy_support", "gtproxy_support_cached", "gtproxy_reliable_query", "gtproxy_rejected_query",
    "gtproxy_hard_background", "gtproxy_ignored_near",
    "gtproxy_support_similarity", "gtproxy_query_similarity",
    "gtproxy_background_similarity", "gtproxy_momentum_projector_update",
    "gtproxy_active_epoch", "gtproxy_support_gathered",
    "dualproxy_loss_raw", "dualproxy_loss_proxy_ce",
    "dualproxy_loss_supcon", "dualproxy_loss_separation",
    "dualproxy_loss_balance", "dualproxy_loss_proxy_ce_weighted",
    "dualproxy_loss_supcon_weighted", "dualproxy_loss_separation_weighted",
    "dualproxy_loss_balance_weighted", "dualproxy_separation_fraction",
    "dualproxy_metric_foreground", "dualproxy_metric_background",
    "dualproxy_fg_similarity_gap",
    "dualproxy_bg_similarity_gap", "dualproxy_gt_support",
    "dualproxy_reliable_query", "dualproxy_rejected_query",
    "dualproxy_ignored_near", "dualproxy_hard_background",
    "dualproxy_random_background",
    "dualproxy_foreground_cached", "dualproxy_background_cached",
    "dualproxy_momentum_projector_update", "dualproxy_active_epoch",
    "dualproxy_foreground_gathered", "dualproxy_background_gathered",
    "prototype_initialization_event", "prototype_epoch_refresh_event",
    "prototype_reinitialization_count", "prototype_update_count",
    "foreground_assignment_share_min", "foreground_effective_prototypes",
    "foreground_pairwise_similarity_max",
    "background_assignment_share_min", "background_effective_prototypes",
    "background_pairwise_similarity_max", "cross_bank_similarity_max",
    "sscp_loss_raw", "sscp_loss_proto_ce", "sscp_loss_margin",
    "sscp_loss_fused_ce", "sscp_loss_proto_ce_weighted",
    "sscp_loss_margin_weighted", "sscp_loss_fused_ce_weighted",
    "sscp_hard_positive", "sscp_random_positive", "sscp_rejected_match",
    "sscp_ignored_near", "sscp_hard_background", "sscp_random_background",
    "sscp_positive_margin", "sscp_background_margin",
    "sscp_alpha_foreground", "sscp_alpha_background", "sscp_balanced_count",
    "sscp_epoch_update", "sscp_foreground_gathered",
    "sscp_background_gathered", "sscp_foreground_assignment_share_min",
    "sscp_background_assignment_share_min",
    "sscp_foreground_effective_prototypes",
    "sscp_background_effective_prototypes",
    "sscp_foreground_pairwise_similarity_max",
    "sscp_background_pairwise_similarity_max", "sscp_cross_bank_similarity_max",
    "sscp_foreground_reinitialized", "sscp_background_reinitialized",
    "potp_loss_raw", "potp_loss_pair", "potp_loss_positive",
    "potp_loss_negative", "potp_loss_balance", "potp_loss_diversity",
    "potp_positive_queries", "potp_gt_support", "potp_hard_background",
    "potp_random_background", "potp_ignored_near",
    "potp_positive_score", "potp_background_score", "potp_score_gap",
    "potp_pair_cosine", "potp_effective_prototypes",
    "potp_assignment_share_min", "potp_shared_gradient_scale",
    "potp_foreground_pairwise_similarity_max",
    "potp_has_background_prototypes", "potp_inference_fusion",
    "potp_initialization_event", "potp_initial_support_cached",
    "pgrp_loss_raw", "pgrp_loss_proto_rank", "pgrp_loss_cls_rank",
    "pgrp_loss_metric", "pgrp_loss_center", "pgrp_loss_balance",
    "pgrp_loss_diversity", "pgrp_reliable_matched",
    "pgrp_rejected_matched", "pgrp_hard_positive", "pgrp_gt_support",
    "pgrp_eligible_background", "pgrp_hard_background",
    "pgrp_random_background", "pgrp_ignored_near",
    "pgrp_raw_positive", "pgrp_raw_background", "pgrp_cls_rank_gap",
    "pgrp_proto_positive", "pgrp_proto_background", "pgrp_proto_rank_gap",
    "pgrp_proto_violation_rate", "pgrp_cls_violation_rate",
    "pgrp_effective_prototypes", "pgrp_assignment_share_min",
    "pgrp_shared_gradient_scale", "pgrp_support_cached",
    "pgrp_initialization_event", "pgrp_refresh_event",
    "pgrp_foreground_pairwise_similarity_max", "pgrp_raw_support_gathered",
    "pgrp_inference_fusion", "prototype_ready",
    "prr_loss_raw", "prr_loss_rescue", "prr_loss_proto_rank",
    "prr_loss_metric", "prr_loss_center", "prr_loss_balance",
    "prr_loss_diversity", "prr_reliable_matched", "prr_rejected_matched",
    "prr_rescue_selected", "prr_metric_positive", "prr_gt_support",
    "prr_eligible_background", "prr_hard_background",
    "prr_random_background", "prr_ignored_near",
    "prr_selected_probability", "prr_selected_score",
    "prr_selected_weight", "prr_selected_margin",
    "prr_proto_positive", "prr_proto_background", "prr_proto_score_gap",
    "prr_effective_prototypes", "prr_assignment_share_min",
    "prr_support_score_low", "prr_support_score_high",
    "prr_detector_gradient_scale", "prr_detector_background_updates",
    "prr_support_cached", "prr_initialization_event", "prr_refresh_event",
    "prr_foreground_pairwise_similarity_max", "prr_raw_support_gathered",
    "prr_inference_fusion",
    "plsp_active", "plsp_selected", "plsp_low_confidence_targets",
    "plsp_unassigned",
    "plsp_local_owner", "plsp_distance_pass", "plsp_similarity_pass",
    "plsp_selected_probability", "plsp_selected_matched_probability",
    "plsp_maximum_matched_probability", "plsp_selected_distance",
    "plsp_distance_improvement", "plsp_similarity_improvement",
    "plsp_positive_weight", "plsp_regression_updates",
    "plsp_matcher_changes",
]


def _write_prototype_epoch_csv(path, epoch, losses, step_diag, state_diag):
    row = {
        "epoch": int(epoch),
        "regression_loss": losses[0],
        "classification_loss": losses[1],
        "weighted_prototype_loss": losses[2],
    }
    row.update(step_diag)
    row.update(state_diag)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PROTOTYPE_EPOCH_CSV_FIELDS, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PROTOTYPE_EPOCH_CSV_FIELDS})


def _print_prototype_health(epoch, step_diag, state_diag):
    def value(source, key, default=float("nan")):
        result = source.get(key, default)
        return default if result is None else result

    if "plsp_selected" in step_diag:
        print(
            "[Prototype-Local-Soft-Positive-health] "
            f"epoch={epoch}, selected={value(step_diag, 'plsp_selected', 0):.2f}, "
            f"low_conf_targets={value(step_diag, 'plsp_low_confidence_targets', 0):.1f}, "
            f"funnel(unassigned/local/distance/sim)="
            f"{value(step_diag, 'plsp_unassigned', 0):.1f}/"
            f"{value(step_diag, 'plsp_local_owner', 0):.1f}/"
            f"{value(step_diag, 'plsp_distance_pass', 0):.1f}/"
            f"{value(step_diag, 'plsp_similarity_pass', 0):.1f}, "
            f"candidate_p/matched_p/dist/delta_dist/delta_sim="
            f"{value(step_diag, 'plsp_selected_probability'):.4f}/"
            f"{value(step_diag, 'plsp_selected_matched_probability'):.4f}/"
            f"{value(step_diag, 'plsp_selected_distance'):.4f}/"
            f"{value(step_diag, 'plsp_distance_improvement'):.4f}/"
            f"{value(step_diag, 'plsp_similarity_improvement'):.4f}, "
            f"weak_weight={value(step_diag, 'plsp_positive_weight'):.3f}, "
            f"matcher/reg={value(step_diag, 'plsp_matcher_changes', -1):.0f}/"
            f"{value(step_diag, 'plsp_regression_updates', -1):.0f}, "
            f"effective/share="
            f"{value(step_diag, 'prr_effective_prototypes'):.2f}/"
            f"{value(step_diag, 'prr_assignment_share_min'):.4f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}",
            flush=True,
        )
        return

    if "prr_loss_raw" in step_diag:
        print(
            "[Prototype-Reliability-Rescue-health] "
            f"epoch={epoch}, raw={value(step_diag, 'prr_loss_raw'):.6f}, "
            f"rescue/proto/metric/center/bal/div="
            f"{value(step_diag, 'prr_loss_rescue'):.4f}/"
            f"{value(step_diag, 'prr_loss_proto_rank'):.4f}/"
            f"{value(step_diag, 'prr_loss_metric'):.4f}/"
            f"{value(step_diag, 'prr_loss_center'):.4f}/"
            f"{value(step_diag, 'prr_loss_balance'):.4f}/"
            f"{value(step_diag, 'prr_loss_diversity'):.4f}, "
            f"reliable/rejected/rescue/metric/bg/random="
            f"{value(step_diag, 'prr_reliable_matched', 0):.1f}/"
            f"{value(step_diag, 'prr_rejected_matched', 0):.1f}/"
            f"{value(step_diag, 'prr_rescue_selected', 0):.1f}/"
            f"{value(step_diag, 'prr_metric_positive', 0):.1f}/"
            f"{value(step_diag, 'prr_hard_background', 0):.1f}/"
            f"{value(step_diag, 'prr_random_background', 0):.1f}, "
            f"p/score/margin="
            f"{value(step_diag, 'prr_selected_probability'):.4f}/"
            f"{value(step_diag, 'prr_selected_score'):.4f}/"
            f"{value(step_diag, 'prr_selected_margin'):.4f}, "
            f"proto_gap={value(step_diag, 'prr_proto_score_gap'):.4f}, "
            f"effective/share="
            f"{value(step_diag, 'prr_effective_prototypes'):.2f}/"
            f"{value(step_diag, 'prr_assignment_share_min'):.4f}, "
            f"q20/q80={value(step_diag, 'prr_support_score_low'):.4f}/"
            f"{value(step_diag, 'prr_support_score_high'):.4f}, "
            f"pair_max={value(state_diag, 'prr_foreground_pairwise_similarity_max'):.4f}, "
            f"detector_bg={value(step_diag, 'prr_detector_background_updates', -1):.0f}, "
            f"grad_scale={value(step_diag, 'prr_detector_gradient_scale'):.3f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}",
            flush=True,
        )
        return

    if "pgrp_loss_raw" in step_diag:
        print(
            "[Prototype-Guided-Ranking-health] "
            f"epoch={epoch}, raw={value(step_diag, 'pgrp_loss_raw'):.6f}, "
            f"proto/cls/metric/center/bal/div="
            f"{value(step_diag, 'pgrp_loss_proto_rank'):.4f}/"
            f"{value(step_diag, 'pgrp_loss_cls_rank'):.4f}/"
            f"{value(step_diag, 'pgrp_loss_metric'):.4f}/"
            f"{value(step_diag, 'pgrp_loss_center'):.4f}/"
            f"{value(step_diag, 'pgrp_loss_balance'):.4f}/"
            f"{value(step_diag, 'pgrp_loss_diversity'):.4f}, "
            f"reliable/rejected/hard_pos/eligible/hard_bg/random="
            f"{value(step_diag, 'pgrp_reliable_matched', 0):.1f}/"
            f"{value(step_diag, 'pgrp_rejected_matched', 0):.1f}/"
            f"{value(step_diag, 'pgrp_hard_positive', 0):.1f}/"
            f"{value(step_diag, 'pgrp_eligible_background', 0):.1f}/"
            f"{value(step_diag, 'pgrp_hard_background', 0):.1f}/"
            f"{value(step_diag, 'pgrp_random_background', 0):.1f}, "
            f"rank_gap_proto/cls="
            f"{value(step_diag, 'pgrp_proto_rank_gap'):.4f}/"
            f"{value(step_diag, 'pgrp_cls_rank_gap'):.4f}, "
            f"violation_proto/cls="
            f"{value(step_diag, 'pgrp_proto_violation_rate'):.4f}/"
            f"{value(step_diag, 'pgrp_cls_violation_rate'):.4f}, "
            f"effective/share="
            f"{value(step_diag, 'pgrp_effective_prototypes'):.2f}/"
            f"{value(step_diag, 'pgrp_assignment_share_min'):.4f}, "
            f"pair_max={value(state_diag, 'pgrp_foreground_pairwise_similarity_max'):.4f}, "
            f"shared_scale={value(step_diag, 'pgrp_shared_gradient_scale'):.3f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}",
            flush=True,
        )
        return

    if "potp_loss_raw" in step_diag:
        print(
            "[Positive-Only-Prototype-health] "
            f"epoch={epoch}, raw={value(step_diag, 'potp_loss_raw'):.6f}, "
            f"pair/pos/neg/bal/div="
            f"{value(step_diag, 'potp_loss_pair'):.4f}/"
            f"{value(step_diag, 'potp_loss_positive'):.4f}/"
            f"{value(step_diag, 'potp_loss_negative'):.4f}/"
            f"{value(step_diag, 'potp_loss_balance'):.4f}/"
            f"{value(step_diag, 'potp_loss_diversity'):.4f}, "
            f"query/support/hard/random/near="
            f"{value(step_diag, 'potp_positive_queries', 0):.1f}/"
            f"{value(step_diag, 'potp_gt_support', 0):.1f}/"
            f"{value(step_diag, 'potp_hard_background', 0):.1f}/"
            f"{value(step_diag, 'potp_random_background', 0):.1f}/"
            f"{value(step_diag, 'potp_ignored_near', 0):.1f}, "
            f"score_pos/bg/gap="
            f"{value(step_diag, 'potp_positive_score'):.4f}/"
            f"{value(step_diag, 'potp_background_score'):.4f}/"
            f"{value(step_diag, 'potp_score_gap'):.4f}, "
            f"pair_cos={value(step_diag, 'potp_pair_cosine'):.4f}, "
            f"effective/share={value(step_diag, 'potp_effective_prototypes'):.2f}/"
            f"{value(step_diag, 'potp_assignment_share_min'):.4f}, "
            f"pair_max={value(state_diag, 'potp_foreground_pairwise_similarity_max'):.4f}, "
            f"shared_scale={value(step_diag, 'potp_shared_gradient_scale'):.3f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}",
            flush=True,
        )
        return

    if "sscp_loss_raw" in step_diag:
        print(
            "[SSCP-health] "
            f"epoch={epoch}, raw={value(step_diag, 'sscp_loss_raw'):.6f}, "
            f"ce/margin/fused="
            f"{value(step_diag, 'sscp_loss_proto_ce'):.4f}/"
            f"{value(step_diag, 'sscp_loss_margin'):.4f}/"
            f"{value(step_diag, 'sscp_loss_fused_ce'):.4f}, "
            f"hard/random_pos={value(step_diag, 'sscp_hard_positive', 0):.1f}/"
            f"{value(step_diag, 'sscp_random_positive', 0):.1f}, "
            f"rejected/near={value(step_diag, 'sscp_rejected_match', 0):.1f}/"
            f"{value(step_diag, 'sscp_ignored_near', 0):.1f}, "
            f"hard/random_bg={value(step_diag, 'sscp_hard_background', 0):.1f}/"
            f"{value(step_diag, 'sscp_random_background', 0):.1f}, "
            f"margin_pos/bg={value(step_diag, 'sscp_positive_margin'):.4f}/"
            f"{value(step_diag, 'sscp_background_margin'):.4f}, "
            f"alpha_fg/bg={value(step_diag, 'sscp_alpha_foreground'):.5f}/"
            f"{value(step_diag, 'sscp_alpha_background'):.5f}, "
            f"effective_fg/bg="
            f"{value(state_diag, 'sscp_foreground_effective_prototypes'):.2f}/"
            f"{value(state_diag, 'sscp_background_effective_prototypes'):.2f}, "
            f"share_fg/bg="
            f"{value(state_diag, 'sscp_foreground_assignment_share_min'):.4f}/"
            f"{value(state_diag, 'sscp_background_assignment_share_min'):.4f}, "
            f"cross={value(state_diag, 'sscp_cross_bank_similarity_max'):.4f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}",
            flush=True,
        )
        return

    if "dualproxy_loss_raw" in step_diag:
        print(
            "[Dual-Proxy-health] "
            f"epoch={epoch}, raw={value(step_diag, 'dualproxy_loss_raw'):.6f}, "
            f"ce/supcon/separation/balance="
            f"{value(step_diag, 'dualproxy_loss_proxy_ce'):.4f}/"
            f"{value(step_diag, 'dualproxy_loss_supcon'):.4f}/"
            f"{value(step_diag, 'dualproxy_loss_separation'):.4f}/"
            f"{value(step_diag, 'dualproxy_loss_balance'):.4f}, "
            f"weighted_ce/supcon/separation/balance="
            f"{value(step_diag, 'dualproxy_loss_proxy_ce_weighted'):.4f}/"
            f"{value(step_diag, 'dualproxy_loss_supcon_weighted'):.4f}/"
            f"{value(step_diag, 'dualproxy_loss_separation_weighted'):.4f}/"
            f"{value(step_diag, 'dualproxy_loss_balance_weighted'):.4f}, "
            f"separation_fraction={value(step_diag, 'dualproxy_separation_fraction'):.4f}, "
            f"metric_fg/bg={value(step_diag, 'dualproxy_metric_foreground', 0):.1f}/"
            f"{value(step_diag, 'dualproxy_metric_background', 0):.1f}, "
            f"support/reliable/rejected/near/hard/random="
            f"{value(step_diag, 'dualproxy_gt_support', 0):.1f}/"
            f"{value(step_diag, 'dualproxy_reliable_query', 0):.1f}/"
            f"{value(step_diag, 'dualproxy_rejected_query', 0):.1f}/"
            f"{value(step_diag, 'dualproxy_ignored_near', 0):.1f}/"
            f"{value(step_diag, 'dualproxy_hard_background', 0):.1f}/"
            f"{value(step_diag, 'dualproxy_random_background', 0):.1f}, "
            f"gap_fg/bg="
            f"{value(step_diag, 'dualproxy_fg_similarity_gap'):.4f}/"
            f"{value(step_diag, 'dualproxy_bg_similarity_gap'):.4f}, "
            f"effective_fg/bg="
            f"{value(state_diag, 'foreground_effective_prototypes'):.2f}/"
            f"{value(state_diag, 'background_effective_prototypes'):.2f}, "
            f"pair_fg/bg/cross="
            f"{value(state_diag, 'foreground_pairwise_similarity_max'):.4f}/"
            f"{value(state_diag, 'background_pairwise_similarity_max'):.4f}/"
            f"{value(state_diag, 'cross_bank_similarity_max'):.4f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}",
            flush=True,
        )
        return

    if "gtproxy_loss_raw" in step_diag:
        print(
            "[GT-Proxy-health] "
            f"epoch={epoch}, raw={value(step_diag, 'gtproxy_loss_raw'):.6f}, "
            f"align/query/bg/balance="
            f"{value(step_diag, 'gtproxy_loss_alignment'):.4f}/"
            f"{value(step_diag, 'gtproxy_loss_query'):.4f}/"
            f"{value(step_diag, 'gtproxy_loss_background'):.4f}/"
            f"{value(step_diag, 'gtproxy_loss_balance'):.4f}, "
            f"support/reliable/rejected/near/far="
            f"{value(step_diag, 'gtproxy_support', 0):.1f}/"
            f"{value(step_diag, 'gtproxy_reliable_query', 0):.1f}/"
            f"{value(step_diag, 'gtproxy_rejected_query', 0):.1f}/"
            f"{value(step_diag, 'gtproxy_ignored_near', 0):.1f}/"
            f"{value(step_diag, 'gtproxy_hard_background', 0):.1f}, "
            f"sim_support/query/bg="
            f"{value(step_diag, 'gtproxy_support_similarity'):.4f}/"
            f"{value(step_diag, 'gtproxy_query_similarity'):.4f}/"
            f"{value(step_diag, 'gtproxy_background_similarity'):.4f}, "
            f"effective_k={value(state_diag, 'foreground_effective_prototypes'):.2f}, "
            f"pair_max={value(state_diag, 'foreground_pairwise_similarity_max'):.4f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}, "
            f"batch_fraction={value(step_diag, 'train_batch_fraction'):.4f}",
            flush=True,
        )
        return

    if "ft_proto_loss_raw" in step_diag:
        print(
            "[FT-Prototype-health] "
            f"epoch={epoch}, raw_loss={value(step_diag, 'ft_proto_loss_raw'):.6f}, "
            f"positive={value(step_diag, 'ft_proto_positive', 0):.2f}, "
            f"used={value(step_diag, 'ft_proto_positive_used', 0):.2f}, "
            f"hard/random={value(step_diag, 'ft_proto_hard_negative', 0):.2f}/"
            f"{value(step_diag, 'ft_proto_random_negative', 0):.2f}, "
            f"sim_pos/neg={value(step_diag, 'ft_proto_positive_similarity_mean'):.4f}/"
            f"{value(step_diag, 'ft_proto_negative_similarity_mean'):.4f}, "
            f"gap={value(step_diag, 'ft_proto_similarity_gap_mean'):.4f}, "
            f"accept={value(step_diag, 'ft_proto_positive_acceptance_rate'):.4f}, "
            f"proto_acc={value(step_diag, 'ft_proto_positive_vs_fixed_negative_accuracy'):.4f}, "
            f"effective_k={value(step_diag, 'ft_proto_effective_foreground_prototypes'):.2f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}, "
            f"fixed={value(state_diag, 'ft_proto_fixed_state_unchanged', 0):.0f}, "
            f"batch_fraction={value(step_diag, 'train_batch_fraction'):.4f}",
            flush=True,
        )
        return

    if "supervised_proto_loss_raw" in step_diag:
        print(
            "[Supervised-Prototype-health] "
            f"epoch={epoch}, raw_loss={value(step_diag, 'supervised_proto_loss_raw'):.6f}, "
            f"positive={value(step_diag, 'supervised_proto_positive', 0):.2f}, "
            f"hard/random={value(step_diag, 'supervised_proto_hard_negative', 0):.2f}/"
            f"{value(step_diag, 'supervised_proto_random_negative', 0):.2f}, "
            f"margin_pos/hard/random="
            f"{value(step_diag, 'supervised_proto_positive_margin'):.4f}/"
            f"{value(step_diag, 'supervised_proto_hard_margin'):.4f}/"
            f"{value(step_diag, 'supervised_proto_random_margin'):.4f}, "
            f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
            f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}, "
            f"fixed={value(state_diag, 'supervised_proto_fixed_state_unchanged', 0):.0f}, "
            f"batch_fraction={value(step_diag, 'train_batch_fraction'):.4f}",
            flush=True,
        )
        return

    print(
        "[Prototype-health] "
        f"epoch={epoch}, raw_loss={value(step_diag, 'prototype_loss_raw'):.6f}, "
        f"pos_margin={value(step_diag, 'prototype_positive_margin_mean'):.6f}, "
        f"bg_margin={value(step_diag, 'prototype_background_margin_mean'):.6f}, "
        f"pos_acc={value(step_diag, 'prototype_positive_correct_rate'):.4f}, "
        f"bg_acc={value(step_diag, 'prototype_background_correct_rate'):.4f}, "
        f"hard/rand_acc={value(step_diag, 'prototype_hard_background_correct_rate'):.4f}/"
        f"{value(step_diag, 'prototype_random_background_correct_rate'):.4f}, "
        f"grad_ratio={value(step_diag, 'gradient_all_proto_cls_ratio'):.4f}, "
        f"grad_cos={value(step_diag, 'gradient_all_cosine'):.4f}, "
        f"effective_fg/bg={value(state_diag, 'foreground_effective_prototypes'):.2f}/"
        f"{value(state_diag, 'background_effective_prototypes'):.2f}, "
        f"cross_max={value(state_diag, 'foreground_background_similarity_max'):.4f}, "
        f"fresh_cross={value(state_diag, 'fresh_foreground_background_similarity_max'):.4f}, "
        f"probe_drift={value(state_diag, 'projector_probe_drift_cosine'):.4f}, "
        f"refresh={value(state_diag, 'prototype_epoch_refresh_events', 0):.0f}, "
        f"refresh_pos/bg={value(state_diag, 'prototype_epoch_refresh_post_positive_correct_rate'):.4f}/"
        f"{value(state_diag, 'prototype_epoch_refresh_post_background_correct_rate'):.4f}, "
        f"refresh_probe_min={value(state_diag, 'prototype_epoch_refresh_projector_drift_min'):.4f}, "
        f"refresh_age_max={value(state_diag, 'prototype_epoch_refresh_center_age_max'):.0f}, "
        f"batch_fraction={value(step_diag, 'train_batch_fraction'):.4f}, "
        f"worst_pair={value(state_diag, 'worst_cross_foreground_index', -1)}/"
        f"{value(state_diag, 'worst_cross_background_index', -1)}, "
        f"pair_share={value(state_diag, 'worst_cross_foreground_assignment_share', 0.0):.3f}/"
        f"{value(state_diag, 'worst_cross_background_assignment_share', 0.0):.3f}, "
        f"reinit_fg/bg={value(state_diag, 'foreground_reinitialized', 0)}/"
        f"{value(state_diag, 'background_reinitialized', 0)}",
        flush=True,
    )


def train():
    if args.distributed:
        rank = args.gpu
    else:
        rank = 0
    # 创建输出目录（如果不存在）
    # 创建输出目录（如果不存在）
    if rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        
        # === 核心修改：判断是否为续训，动态决定写入模式 ===
        write_mode = 'a' if args.resume else 'w'
        with open(log_file, write_mode) as f:
            if not args.resume:
                f.write("Training Log\n")
            else:
                f.write(f"\n\n{'='*40}\n")
                f.write(f"--- 触发断点续训: 从 {args.resume} 恢复 ---\n")
                f.write(f"{'='*40}\n")
        # #run = wandb.init(project='wandb_usage', entity="zhouzh")
        # run.name = run.id
        # run.save()
        # cfg = wandb.config
        for k, v in args.__dict__.items():
            # setattr(cfg, k, v)
            pass

    model = build_model(args).cuda(rank)
    model_without_ddp = model
    validate_frozen_teacher_initialization(args, model_without_ddp)
    _freeze_detector_for_proxy_gate(
        model_without_ddp, getattr(args, 'proto_freeze_detector', 0)
    )

    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=bool(getattr(args, 'proto_enable', False)),
        )
        model_without_ddp = model.module

    prototype_head_before_load = get_prototype_head(model_without_ddp)
    fixed_bank_hash_before_load = (
        prototype_head_before_load.fixed_state_sha256()
        if _should_protect_fixed_prototype_bank(args)
        and prototype_head_before_load is not None
        and hasattr(prototype_head_before_load, 'fixed_state_sha256')
        else ''
    )

    matcher = build_matcher(args)
    criterion = build_criterion(rank, matcher, args)
    trainable_parameters = [
        parameter for parameter in model_without_ddp.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError('optimizer has no trainable parameters')
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )

    data_loaders = construct_dataset()
    evaluator = Evaluator(data_loaders['test'])
    first_eval = True

    if args.resume and args.init_checkpoint:
        raise ValueError('--resume and --init_checkpoint are mutually exclusive')

    # Resume restores optimizer/epoch. init_checkpoint only initializes compatible weights.
    if args.init_checkpoint:
        checkpoint_report = (
            os.path.join(args.output_dir, "checkpoint_initialization_report.json")
            if rank == 0 and args.output_dir
            else ""
        )
        load_model_weights(
            args.init_checkpoint, model_without_ddp, report_path=checkpoint_report
        )
    metrics = load_checkpoint(args, model_without_ddp, optimizer) if args.resume else {}
    if fixed_bank_hash_before_load:
        fixed_bank_hash_after_load = get_prototype_head(
            model_without_ddp
        ).fixed_state_sha256()
        if fixed_bank_hash_after_load != fixed_bank_hash_before_load:
            raise RuntimeError(
                'fixed prototype bank changed while loading checkpoint: '
                f'before={fixed_bank_hash_before_load}, '
                f'after={fixed_bank_hash_after_load}'
            )
    if int(getattr(args, "reset_rng_after_init", 0)):
        reset_seed = int(args.seed) + int(get_rank())
        _reset_all_rng(reset_seed)
        print(
            f"[RNG-reset] rank={get_rank()}, seed={reset_seed}, "
            f"fingerprint={_rng_fingerprint(rank)}",
            flush=True,
        )
    if rank == 0 and args.output_dir:
        with open(
            os.path.join(args.output_dir, "run_initialization.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "seed": int(args.seed),
                    "reset_rng_after_init": int(args.reset_rng_after_init),
                    "rng_fingerprint": _rng_fingerprint(rank),
                    "init_checkpoint": args.init_checkpoint,
                    "resume": args.resume,
                    "prototype_enabled": bool(args.proto_enable),
                    "prototype_mode": str(getattr(args, "proto_mode", "legacy_online")),
                    "prototype_bank_path": str(getattr(args, "proto_bank_path", "")),
                    "prototype_bank_id": getattr(
                        get_prototype_head(model_without_ddp), "bank_id", ""
                    ),
                    "prototype_fixed_state_sha256": (
                        get_prototype_head(model_without_ddp).fixed_state_sha256()
                        if get_prototype_head(model_without_ddp) is not None
                        and hasattr(
                            get_prototype_head(model_without_ddp),
                            "fixed_state_sha256",
                        )
                        else ""
                    ),
                    "prototype_metadata": getattr(
                        model_without_ddp, "prototype_metadata", {}
                    ),
                    "optimizer_projector_only": bool(
                        getattr(model_without_ddp, "optimizer_projector_only", False)
                    ),
                    "parameter_groups": _parameter_group_summary(
                        model_without_ddp, include_hash=True
                    ),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    max_cls_mf1 = metrics.get('分类F1', 0)

    # 初始化损失记录列表
    reg_losses =[]
    cls_losses = []
    proto_losses = []
    total_losses =[]

    last_completed_epoch = None
    for epoch in range(args.start_epoch, args.epochs):
        # train_one_epoch(model, data_loaders['train'], optimizer, args.clip_max_norm, epoch, criterion, rank)

        # 训练一个epoch并获取损失
        prototype_head = get_prototype_head(model_without_ddp)
        prototype_active = (
            prototype_head is not None and epoch >= int(getattr(args, 'proto_start_epoch', 0))
        )
        if prototype_active:
            if hasattr(prototype_head, 'set_epoch'):
                prototype_head.set_epoch(epoch)
            prototype_head.begin_epoch()

        rng_start = _rng_fingerprint(rank)

        reg_tl, cls_tl, proto_tl, prototype_step_diag = train_one_epoch(
            model,
            data_loaders['train'],
            optimizer,
            args.clip_max_norm,
            epoch,
            criterion,
            rank,
        )
        last_completed_epoch = epoch

        prototype_epoch_diag = {}
        if prototype_active:
            prototype_epoch_diag = prototype_head.finalize_epoch()
            if rank == 0:
                _print_prototype_health(
                    epoch, prototype_step_diag, prototype_epoch_diag
                )

        dataset_debug = _collect_dataset_debug(data_loaders['train'].dataset, epoch)
        rng_end = _rng_fingerprint(rank)

        # 记录损失
        reg_losses.append(reg_tl)
        cls_losses.append(cls_tl)
        proto_losses.append(proto_tl)
        total_losses.append(reg_tl + cls_tl + proto_tl)


        # 写入训练损失到文件
        if rank == 0 and args.output_dir:
            log_file = os.path.join(args.output_dir, 'training_log.txt')
            with open(log_file, 'a') as f:
                compact_prototype_state = {
                    key: value for key, value in prototype_epoch_diag.items()
                    if not isinstance(value, list)
                }
                f.write(
                    f"Epoch {epoch}: Regression Loss={reg_tl:.4f}, "
                    f"Classification Loss={cls_tl:.4f}, Prototype Loss={proto_tl:.4f}, "
                    f"Total Loss={reg_tl + cls_tl + proto_tl:.4f}, "
                    f"Prototype State={compact_prototype_state}\n"
                )
            _append_jsonl(
                os.path.join(args.output_dir, "run_debug.jsonl"),
                {
                    "epoch": epoch,
                    "rng_start": rng_start,
                    "rng_end": rng_end,
                    "learning_rates": [
                        float(group["lr"]) for group in optimizer.param_groups
                    ],
                    "losses": {
                        "regression": reg_tl,
                        "classification": cls_tl,
                        "weighted_prototype": proto_tl,
                    },
                    "optimizer": {
                        key: value for key, value in prototype_step_diag.items()
                        if key.startswith("optimizer_")
                    },
                    "parameter_groups": _parameter_group_summary(model_without_ddp),
                    "dataset_ranks": dataset_debug,
                },
            )
            if prototype_active:
                debug_path = os.path.join(args.output_dir, 'prototype_debug.jsonl')
                _append_jsonl(
                    debug_path,
                    {
                        'epoch': epoch,
                        'step_mean': prototype_step_diag,
                        'prototype_state': prototype_epoch_diag,
                    },
                )
                _write_prototype_epoch_csv(
                    os.path.join(args.output_dir, "prototype_epoch_metrics.csv"),
                    epoch,
                    (reg_tl, cls_tl, proto_tl),
                    prototype_step_diag,
                    prototype_epoch_diag,
                )

        if trigger_eval(epoch, args.start_eval):
            if first_eval:
                save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='best')
                first_eval = False

            metrics = evaluator.calculate_metrics(model, rank=rank, effective_matching_dis=args.match_dis)
            cls_mf1 = metrics['分类指标'][-1]
            print(metrics)

            # save most recent epoch model
            save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='recent')

            # cls_mf1 = sum(metrics['分类F1'][:2]) / 2
            # print(metrics, cls_mf1)
            
            # 写入评估指标到文件
            if rank == 0 and args.output_dir:
                log_file = os.path.join(args.output_dir, 'training_log.txt')
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch} Evaluation:\n")
                    f.write(f"  Detection Precision={metrics['检测指标'][0]:.4f}, Recall={metrics['检测指标'][1]:.4f}, F1={metrics['检测指标'][2]:.4f}\n")
                    f.write(f"  Classification Precision={metrics['分类指标'][0]:.4f}, Recall={metrics['分类指标'][1]:.4f}, F1={metrics['分类指标'][2]:.4f}\n")
                    f.write(f"  Best Classification F1: {max_cls_mf1:.4f}\n")

            if rank == 0:
                # wandb.log(dict(zip(["检测精度", "检测召回", "检测F1"], metrics['检测指标'])))
                # wandb.log(dict(zip(["分类精度", "分类召回", "分类F1"], metrics['分类指标'])))
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.plot(reg_losses, label='Regression Loss')
                plt.plot(cls_losses, label='Classification Loss')
                plt.plot(proto_losses, label='Weighted Prototype Loss')
                plt.plot(total_losses, label='Total Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Curve')
                plt.legend()

                # 保存到输出目录
                loss_curve_path = os.path.join(args.output_dir, 'loss_curve.png')
                plt.savefig(loss_curve_path)
                print(f"Loss curve saved to {loss_curve_path}")

                if max_cls_mf1 < cls_mf1:
                    max_cls_mf1 = cls_mf1
                    if args.output_dir:
                        save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='best')
                print("max cls_mf1:", max_cls_mf1)
                
        periodic_interval = int(getattr(args, 'checkpoint_interval', 0))
        explicit_periodic_save = should_save_checkpoint(epoch, periodic_interval)
        legacy_periodic_save = (
            periodic_interval <= 0 and epoch >= 30 and (epoch - 30) % 20 == 0
        )
        if rank == 0 and (explicit_periodic_save or legacy_periodic_save):
            current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            label = checkpoint_label(epoch) if explicit_periodic_save else epoch
            model_name = f"model_epoch_{label}_{current_time}.pth"
            model_path = os.path.join(args.output_dir, model_name)
            torch.save({
        'epoch': epoch,
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args),
    }, model_path)
            print(f"Model saved to {model_path}")
            if args.output_dir:
                log_file = os.path.join(args.output_dir, 'training_log.txt')
                with open(log_file, 'a') as f:
                    f.write(f"Model saved: {model_path}\n")

    if (
        rank == 0
        and int(getattr(args, 'debug_save_final_checkpoint', 0))
        and last_completed_epoch is not None
        and args.output_dir
    ):
        save_model(
            last_completed_epoch,
            args,
            metrics,
            model_without_ddp,
            optimizer,
            mode='recent',
        )
        print(
            f'[Debug-final-checkpoint] epoch={last_completed_epoch}, '
            f'path={os.path.join(os.path.abspath(args.output_dir), "recent_model.pth")}',
            flush=True,
        )

    if args.distributed:
        cleanup()


def test():
    # 统一设置根目录变量，方便后续修改
    base_dir = 'pth/0326_bd'
    
    data_loaders = construct_dataset()
    model = build_model(args).cuda()
    
    # 模型加载路径
    model_path = os.path.join(base_dir, 'best_model.pth')
    # 修改点：使用 strict=False 忽略多余的键（如 proto_enhance.*）
    model.load_state_dict(torch.load(model_path)['model'], strict=False)
    model.eval()
    
    evaluator = Evaluator(data_loaders['test'])
    
    # 1. 可视化图片输出路径：pth/0326_bd/vis
    vis_dir = os.path.join(base_dir, 'vis')
    evaluator.visual_analysis(model, output_dir=vis_dir)
    
    # 计算测试指标
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)
    print("测试指标:")
    print(f"检测指标: 精度={metrics['检测指标'][0]:.4f}, 召回={metrics['检测指标'][1]:.4f}, F1={metrics['检测指标'][2]:.4f}")
    print(f"分类指标: 精度={metrics['分类指标'][0]:.4f}, 召回={metrics['分类指标'][1]:.4f}, F1={metrics['分类指标'][2]:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # 处理文件路径不存在的情况（写txt操作）
    # import os
    file_path = os.path.join(base_dir, 'test_results.txt')
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)   # 如果 base_dir 不存在也会自动创建
    
    with open(file_path, 'w') as f:
        f.write("=== 测试结果 ===\n")
        f.write(f"检测指标: {metrics['检测指标']}\n")
        f.write(f"分类指标: {metrics['分类指标']}\n")
        f.write(f"MSE: {metrics['MSE']}\n")
        f.write(f"MAE: {metrics['MAE']}\n")


def get_prototype_head(model):
    base_model = model.module if hasattr(model, 'module') else model
    return getattr(base_model, 'prototype_head', None)


def get_prototype_debug_tensors(model):
    base_model = model.module if hasattr(model, 'module') else model
    getter = getattr(base_model, 'get_prototype_debug_tensors', None)
    return getter() if getter is not None else None


def _gradient_pair_metrics(primary_loss, auxiliary_loss, named_tensors):
    tensors = [tensor for _, tensor in named_tensors]
    primary_gradients = torch.autograd.grad(
        primary_loss,
        tensors,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        tensors,
        retain_graph=True,
        allow_unused=True,
    )
    diagnostics = {}
    total_dot = primary_loss.new_zeros(())
    total_primary_sq = primary_loss.new_zeros(())
    total_auxiliary_sq = primary_loss.new_zeros(())
    for (name, _), primary_grad, auxiliary_grad in zip(
        named_tensors, primary_gradients, auxiliary_gradients
    ):
        diagnostics[f"gradient_{name}_primary_available"] = float(
            primary_grad is not None
        )
        diagnostics[f"gradient_{name}_auxiliary_available"] = float(
            auxiliary_grad is not None
        )
        if primary_grad is None or auxiliary_grad is None:
            diagnostics[f"gradient_{name}_available"] = 0.0
            continue
        primary_detached = primary_grad.detach().float()
        auxiliary_detached = auxiliary_grad.detach().float()
        dot = (primary_detached * auxiliary_detached).sum()
        primary_sq = primary_detached.square().sum()
        auxiliary_sq = auxiliary_detached.square().sum()
        primary_norm = primary_sq.sqrt()
        auxiliary_norm = auxiliary_sq.sqrt()
        cosine = dot / (primary_norm * auxiliary_norm + 1e-12)
        diagnostics[f"gradient_{name}_available"] = 1.0
        diagnostics[f"gradient_{name}_cls_norm"] = float(primary_norm.item())
        diagnostics[f"gradient_{name}_proto_norm"] = float(auxiliary_norm.item())
        diagnostics[f"gradient_{name}_proto_cls_ratio"] = float(
            (auxiliary_norm / (primary_norm + 1e-12)).item()
        )
        diagnostics[f"gradient_{name}_cosine"] = float(cosine.item())
        total_dot = total_dot + dot
        total_primary_sq = total_primary_sq + primary_sq
        total_auxiliary_sq = total_auxiliary_sq + auxiliary_sq

    total_primary_norm = total_primary_sq.sqrt()
    total_auxiliary_norm = total_auxiliary_sq.sqrt()
    diagnostics["gradient_all_cls_norm"] = float(total_primary_norm.item())
    diagnostics["gradient_all_proto_norm"] = float(total_auxiliary_norm.item())
    diagnostics["gradient_all_proto_cls_ratio"] = float(
        (total_auxiliary_norm / (total_primary_norm + 1e-12)).item()
    )
    diagnostics["gradient_all_cosine"] = float(
        (total_dot / (total_primary_norm * total_auxiliary_norm + 1e-12)).item()
    )
    return diagnostics


def _save_prototype_failure_snapshot(
    epoch, step, error, outputs, targets, output_dir
):
    if not output_dir:
        return ""
    debug_dir = os.path.join(output_dir, "prototype_debug_failures")
    os.makedirs(debug_dir, exist_ok=True)
    stem = f"epoch{epoch:03d}_step{step:05d}"
    tensor_path = os.path.join(debug_dir, stem + ".pt")
    json_path = os.path.join(debug_dir, stem + ".json")
    payload = {
        "pnt_coords": outputs.get("pnt_coords", torch.empty(0))[:1].detach().cpu(),
        "raw_cls_logits": outputs.get("raw_cls_logits", torch.empty(0))[:1].detach().cpu(),
        "anchor_points": outputs.get("anchor_points", torch.empty(0))[:1].detach().cpu(),
        "gt_points": [item.detach().cpu() for item in targets.get("gt_points", [])[:1]],
        "gt_labels": [item.detach().cpu() for item in targets.get("gt_labels", [])[:1]],
    }
    torch.save(payload, tensor_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"epoch": epoch, "step": step, "error": repr(error), "snapshot": tensor_path},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return tensor_path


def train_one_epoch(model, train_loader, optimizer, max_norm, epoch, criterion, rank):
    model.train()
    _apply_projector_only_train_mode(model)
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = train_loader
    if rank == 0:
        time_string = time.strftime('[%D-%H:%M:%S]', time.localtime())
        iterator = tqdm(train_loader, file=sys.stdout)
        iterator.set_description(f"{time_string} Train epoch-{epoch}")

    reg_tl = cls_tl = proto_tl = 0
    proto_diag_sums = {}
    proto_diag_counts = {}
    processed_steps = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    clipped_steps = 0
    for data_iter_step, (images, points, labels, lengths) in enumerate(iterator):
        max_debug_batches = int(getattr(args, 'debug_max_train_batches', 0))
        if max_debug_batches > 0 and data_iter_step >= max_debug_batches:
            break
        processed_steps += 1
        # warmup lr
        adjust_learning_rate(args,
                             optimizer,
                             data_iter_step / len(iterator) + epoch,
                             rank)

        images = images.cuda(rank)
        points = points.cuda(rank)
        labels = labels.cuda(rank)

        # all points N×2
        targets = {'gt_nums': lengths,
                   'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                   'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

        cell_ratios = [[(targets['gt_labels'][i] == c).sum().item() for c in range(4)] for i in range(images.size(0))]
        cell_ratios = torch.tensor(cell_ratios, dtype=torch.float).cuda(rank)
        cell_ratios = cell_ratios / cell_ratios.sum(1).unsqueeze(-1)
        targets['cell_ratios'] = cell_ratios
        prototype_head = get_prototype_head(model)
        prototype_mode = str(getattr(args, 'proto_mode', 'legacy_online'))
        prototype_active = (
            prototype_head is not None
            and epoch >= int(getattr(args, 'proto_start_epoch', 0))
        )
        prototype_support_points = (
            targets['gt_points']
            if prototype_active and prototype_mode in {
                'gt_foreground_proxy', 'discriminative_dual_proxy',
                'positive_only_train_proto', 'prototype_guided_ranking',
                'prototype_reliability_rescue',
                'prototype_local_soft_positive',
            }
            else None
        )
        outputs = model(
            images,
            prototype_support_points=prototype_support_points,
        )
        indices = criterion.matcher(outputs, targets)
        pre_criterion_proto_diag = {}
        if prototype_active and prototype_mode == 'prototype_local_soft_positive':
            soft_positive_indices, pre_criterion_proto_diag = (
                prototype_head.select_local_soft_positives(
                    outputs['pnt_coords'],
                    outputs['raw_cls_logits'],
                    outputs['proto_embeddings'],
                    outputs['proto_support_embeddings'],
                    outputs['proto_support_valid_mask'],
                    targets,
                    indices,
                )
            )
            outputs['prototype_soft_positive_indices'] = soft_positive_indices
            outputs['prototype_soft_positive_weight'] = (
                prototype_head.soft_positive_weight
            )
        losses = criterion(outputs, targets, indices=indices)
        loss = losses.sum()

        if data_iter_step == 0 and rank == 0 and args.output_dir:
            _append_jsonl(
                os.path.join(args.output_dir, "first_batch_debug.jsonl"),
                {
                    "epoch": int(epoch),
                    "images": _tensor_fingerprint(images),
                    "points": _tensor_fingerprint(points),
                    "labels": _tensor_fingerprint(labels),
                    "pnt_coords": _tensor_fingerprint(outputs["pnt_coords"]),
                    "raw_cls_logits": _tensor_fingerprint(outputs["raw_cls_logits"]),
                    "base_losses": [float(value.item()) for value in losses.detach()],
                    "matcher_source_indices": [
                        _tensor_fingerprint(pair[0]) for pair in indices
                    ],
                    "matcher_target_indices": [
                        _tensor_fingerprint(pair[1]) for pair in indices
                    ],
                },
            )

        weighted_proto_loss = loss * 0.0
        if prototype_active:
            try:
                if prototype_mode in {
                    'frozen_teacher_fg', 'frozen_supervised_metric'
                }:
                    raw_proto_loss, proto_diag = prototype_head.compute_loss(
                        outputs['proto_embeddings'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                    )
                elif prototype_mode == 'discriminative_dual_proxy':
                    raw_proto_loss, proto_diag = prototype_head.compute_loss_and_cache(
                        outputs['proto_embeddings'],
                        outputs['proto_momentum_embeddings'],
                        outputs['proto_support_embeddings'],
                        outputs['proto_momentum_support_embeddings'],
                        outputs['proto_support_valid_mask'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                    )
                elif prototype_mode == 'gt_foreground_proxy':
                    raw_proto_loss, proto_diag = prototype_head.compute_loss_and_cache(
                        outputs['proto_embeddings'],
                        outputs['proto_support_embeddings'],
                        outputs['proto_momentum_support_embeddings'],
                        outputs['proto_support_valid_mask'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                    )
                elif prototype_mode in {
                    'prototype_guided_ranking', 'prototype_reliability_rescue',
                    'prototype_local_soft_positive',
                }:
                    raw_proto_loss, proto_diag = prototype_head.compute_loss(
                        outputs['proto_embeddings'],
                        outputs['proto_support_embeddings'],
                        outputs['proto_candidate_raw_features'],
                        outputs['proto_support_raw_features'],
                        outputs['proto_support_valid_mask'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                    )
                elif prototype_mode == 'positive_only_train_proto':
                    raw_proto_loss, proto_diag = prototype_head.compute_loss(
                        outputs['proto_embeddings'],
                        outputs['proto_support_embeddings'],
                        outputs['proto_support_valid_mask'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                    )
                elif prototype_mode == 'source_supervised_candidate_proto':
                    raw_proto_loss, proto_diag = prototype_head.compute_loss_and_cache(
                        outputs['proto_embeddings'],
                        outputs['proto_fused_logits'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                    )
                else:
                    raw_proto_loss, proto_diag = prototype_head.compute_loss_and_cache(
                        outputs['proto_embeddings'],
                        outputs['pnt_coords'],
                        outputs['raw_cls_logits'],
                        targets,
                        indices,
                        anchor_points=outputs['anchor_points'],
                        source_features=outputs.get('proto_source_features'),
                    )
                if not torch.isfinite(raw_proto_loss):
                    raise FloatingPointError("non-finite prototype loss")
            except Exception as error:
                snapshot = ""
                if rank == 0 and int(getattr(args, 'proto_debug_fail_fast', 1)):
                    snapshot = _save_prototype_failure_snapshot(
                        epoch,
                        data_iter_step,
                        error,
                        outputs,
                        targets,
                        args.output_dir,
                    )
                    print(
                        f"[Prototype-failure] epoch={epoch}, step={data_iter_step}, "
                        f"snapshot={snapshot or '<disabled>'}, error={error!r}",
                        flush=True,
                    )
                raise
            proto_diag.update(pre_criterion_proto_diag)
            weighted_proto_loss = raw_proto_loss * float(args.proto_loss_weight)
            loss = loss + weighted_proto_loss
            debug_interval = int(getattr(args, 'proto_debug_interval', 0))
            if (
                debug_interval > 0
                and bool(prototype_head.prototype_ready.item())
                and data_iter_step % debug_interval == 0
            ):
                internal_debug = get_prototype_debug_tensors(model)
                if internal_debug is None:
                    raise RuntimeError(
                        "prototype debug tensors are unavailable; the model must "
                        "expose its pre-DDP shared feature tensors"
                    )
                all_named_tensors = [('cls_features', internal_debug['cls_features'])]
                all_named_tensors.extend(
                    (f'fpn_{idx}', feature)
                    for idx, feature in enumerate(internal_debug['fpn_features'])
                )
                diagnostic_event = data_iter_step // debug_interval
                named_tensors = select_gradient_probe(
                    all_named_tensors, diagnostic_event
                )
                probe_name = named_tensors[0][0]
                proto_diag.update(
                    _gradient_pair_metrics(losses[1], weighted_proto_loss, named_tensors)
                )
                proto_diag[f'gradient_probe_{probe_name}'] = 1.0
                if rank == 0:
                    print(
                        "[Prototype-gradient] "
                        f"epoch={epoch}, step={data_iter_step}, "
                        f"probe={probe_name}, "
                        f"ratio={proto_diag['gradient_all_proto_cls_ratio']:.6f}, "
                        f"cos={proto_diag['gradient_all_cosine']:.6f}",
                        flush=True,
                    )
            if (
                rank == 0
                and debug_interval > 0
                and data_iter_step % debug_interval == 0
                and args.output_dir
            ):
                step_payload = {
                    "epoch": int(epoch),
                    "step": int(data_iter_step),
                    **proto_diag,
                }
                _append_jsonl(
                    os.path.join(args.output_dir, "prototype_step_debug.jsonl"),
                    step_payload,
                )
                if 'plsp_selected' in proto_diag:
                    step_prefix = "[Prototype-Local-Soft-Positive-step-debug] "
                elif 'prr_loss_raw' in proto_diag:
                    step_prefix = "[Prototype-Reliability-Rescue-step-debug] "
                elif 'pgrp_loss_raw' in proto_diag:
                    step_prefix = "[Prototype-Guided-Ranking-step-debug] "
                elif 'potp_loss_raw' in proto_diag:
                    step_prefix = "[Positive-Only-Prototype-step-debug] "
                elif 'sscp_loss_raw' in proto_diag:
                    step_prefix = "[SSCP-step-debug] "
                elif 'dualproxy_loss_raw' in proto_diag:
                    step_prefix = "[Dual-Proxy-step-debug] "
                elif 'gtproxy_loss_raw' in proto_diag:
                    step_prefix = "[GT-Proxy-step-debug] "
                elif 'ft_proto_loss_raw' in proto_diag:
                    step_prefix = "[FT-Prototype-step-debug] "
                else:
                    step_prefix = "[Prototype-step-debug] "
                print(
                    f"{step_prefix}"
                    f"epoch={epoch}, step={data_iter_step}, "
                    f"ready={proto_diag.get('prototype_ready', 0):.0f}, "
                    f"positive={proto_diag.get('prr_reliable_matched', proto_diag.get('pgrp_hard_positive', proto_diag.get('potp_positive_queries', proto_diag.get('sscp_hard_positive', 0) + proto_diag.get('sscp_random_positive', proto_diag.get('gtproxy_reliable_query', proto_diag.get('ft_proto_positive', proto_diag.get('prototype_positive', 0))))))):.0f}, "
                    f"rejected={proto_diag.get('prr_rejected_matched', proto_diag.get('pgrp_rejected_matched', proto_diag.get('sscp_rejected_match', proto_diag.get('gtproxy_rejected_query', proto_diag.get('ft_proto_rejected_positive', proto_diag.get('prototype_rejected_positive', 0)))))):.0f}, "
                    f"hard/random={proto_diag.get('prr_hard_background', proto_diag.get('pgrp_hard_background', proto_diag.get('potp_hard_background', proto_diag.get('sscp_hard_background', proto_diag.get('gtproxy_hard_background', proto_diag.get('ft_proto_hard_negative', proto_diag.get('prototype_hard_background', 0))))))):.0f}/"
                    f"{proto_diag.get('prr_random_background', proto_diag.get('pgrp_random_background', proto_diag.get('potp_random_background', proto_diag.get('sscp_random_background', proto_diag.get('ft_proto_random_negative', proto_diag.get('prototype_random_background', 0)))))):.0f}, "
                    f"sim_pos={proto_diag.get('prr_proto_positive', proto_diag.get('pgrp_proto_positive', proto_diag.get('potp_positive_score', proto_diag.get('gtproxy_query_similarity', proto_diag.get('ft_proto_positive_similarity_mean', proto_diag.get('prototype_positive_margin_mean', float('nan'))))))):.4f}, "
                    f"sim_neg={proto_diag.get('prr_proto_background', proto_diag.get('pgrp_proto_background', proto_diag.get('potp_background_score', proto_diag.get('gtproxy_background_similarity', proto_diag.get('ft_proto_negative_similarity_mean', proto_diag.get('prototype_background_margin_mean', float('nan'))))))):.4f}, "
                    f"gap={proto_diag.get('prr_proto_score_gap', proto_diag.get('pgrp_proto_rank_gap', proto_diag.get('potp_score_gap', proto_diag.get('ft_proto_similarity_gap_mean', float('nan'))))):.4f}, "
                    f"effective_k={proto_diag.get('prr_effective_prototypes', proto_diag.get('pgrp_effective_prototypes', proto_diag.get('potp_effective_prototypes', proto_diag.get('ft_proto_effective_foreground_prototypes', float('nan'))))):.2f}",
                    flush=True,
                )
            accumulate_finite_diagnostics(
                proto_diag_sums, proto_diag_counts, proto_diag
            )
        loss.backward()

        if max_norm > 0:  # clip gradient
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
            )
            grad_norm_sum += total_grad_norm
            grad_norm_max = max(grad_norm_max, total_grad_norm)
            clipped_steps += int(total_grad_norm > max_norm)
        optimizer.step()
        optimizer.zero_grad()
        if prototype_active and str(getattr(args, 'proto_mode', 'legacy_online')) != 'frozen_teacher_fg':
            refresh_diag = prototype_head.maybe_refresh_prototypes()
            accumulate_finite_diagnostics(
                proto_diag_sums, proto_diag_counts, refresh_diag
            )
            if rank == 0 and int(refresh_diag.get('prototype_refresh_event', 0)):
                refresh_payload = {
                    'epoch': int(epoch),
                    'step': int(data_iter_step),
                    **refresh_diag,
                }
                if args.output_dir:
                    _append_jsonl(
                        os.path.join(
                            args.output_dir, 'prototype_refresh_debug.jsonl'
                        ),
                        refresh_payload,
                    )
                print(
                    "[Prototype-refresh] "
                    f"epoch={epoch}, step={data_iter_step}, "
                    f"count={refresh_diag.get('prototype_refresh_count', 0):.0f}, "
                    f"age={refresh_diag.get('prototype_center_age_steps_before_refresh', 0):.0f}, "
                    f"samples={refresh_diag.get('prototype_refresh_foreground_gathered', 0):.0f}/"
                    f"{refresh_diag.get('prototype_refresh_background_gathered', 0):.0f}, "
                    f"pre/post-pos={refresh_diag.get('prototype_refresh_pre_positive_correct_rate', float('nan')):.4f}/"
                    f"{refresh_diag.get('prototype_refresh_post_positive_correct_rate', float('nan')):.4f}, "
                    f"pre/post-bg={refresh_diag.get('prototype_refresh_pre_background_correct_rate', float('nan')):.4f}/"
                    f"{refresh_diag.get('prototype_refresh_post_background_correct_rate', float('nan')):.4f}, "
                    f"projector-cos={refresh_diag.get('prototype_refresh_projector_drift_cosine', float('nan')):.4f}",
                    flush=True,
                )
        report_losses = torch.cat([losses.detach(), weighted_proto_loss.detach().reshape(1)])
        report_losses /= args.world_size

        if args.distributed:
            dist.all_reduce(report_losses, op=dist.ReduceOp.SUM)
        reg_tl += report_losses[0].item()
        cls_tl += report_losses[1].item()
        proto_tl += report_losses[2].item()

    if processed_steps > 0:
        available_steps = len(train_loader)
        proto_diag_sums["train_batches_processed"] = float(processed_steps)
        proto_diag_counts["train_batches_processed"] = 1
        proto_diag_sums["train_batches_available"] = float(available_steps)
        proto_diag_counts["train_batches_available"] = 1
        proto_diag_sums["train_batch_fraction"] = float(processed_steps) / max(
            available_steps, 1
        )
        proto_diag_counts["train_batch_fraction"] = 1
        proto_diag_sums["optimizer_total_grad_norm_before_clip"] = grad_norm_sum
        proto_diag_counts["optimizer_total_grad_norm_before_clip"] = processed_steps
        proto_diag_sums["optimizer_clip_rate"] = float(clipped_steps)
        proto_diag_counts["optimizer_clip_rate"] = processed_steps
        proto_diag_sums["optimizer_total_grad_norm_max"] = grad_norm_max
        proto_diag_counts["optimizer_total_grad_norm_max"] = 1

    if args.distributed:
        gathered_proto_diagnostics = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered_proto_diagnostics,
            {"sums": proto_diag_sums, "counts": proto_diag_counts},
        )
        proto_diag_sums = {}
        proto_diag_counts = {}
        for rank_payload in gathered_proto_diagnostics:
            for key, value in rank_payload["sums"].items():
                proto_diag_sums[key] = proto_diag_sums.get(key, 0.0) + float(value)
            for key, value in rank_payload["counts"].items():
                proto_diag_counts[key] = proto_diag_counts.get(key, 0) + int(value)

    mean_diag = (
        {
            key: value / proto_diag_counts[key]
            for key, value in proto_diag_sums.items()
        }
        if proto_diag_counts
        else {}
    )

    if rank == 0:
        print(
            f'回归损失: {reg_tl}, 分类损失: {cls_tl}, 加权原型损失: {proto_tl}',
            flush=True,
        )
        if proto_diag_counts:
            if "plsp_selected" in mean_diag:
                prefix = "[Prototype-Local-Soft-Positive-step-mean]"
            elif "prr_loss_raw" in mean_diag:
                prefix = "[Prototype-Reliability-Rescue-step-mean]"
            elif "pgrp_loss_raw" in mean_diag:
                prefix = "[Prototype-Guided-Ranking-step-mean]"
            elif "potp_loss_raw" in mean_diag:
                prefix = "[Positive-Only-Prototype-step-mean]"
            elif "dualproxy_loss_raw" in mean_diag:
                prefix = "[Dual-Proxy-step-mean]"
            elif "gtproxy_loss_raw" in mean_diag:
                prefix = "[GT-Proxy-step-mean]"
            elif "ft_proto_loss_raw" in mean_diag:
                prefix = "[FT-Prototype-step-mean]"
            elif "supervised_proto_loss_raw" in mean_diag:
                prefix = "[Supervised-Prototype-step-mean]"
            else:
                prefix = "[Prototype-step-mean]"
            print(
                f"{prefix} "
                f"raw_loss={mean_diag.get('prr_loss_raw', mean_diag.get('pgrp_loss_raw', mean_diag.get('potp_loss_raw', mean_diag.get('dualproxy_loss_raw', mean_diag.get('gtproxy_loss_raw', mean_diag.get('supervised_proto_loss_raw', mean_diag.get('ft_proto_loss_raw', mean_diag.get('prototype_loss_raw', float('nan'))))))))) :.6f}, "
                f"positive={mean_diag.get('prr_reliable_matched', mean_diag.get('pgrp_hard_positive', mean_diag.get('potp_positive_queries', mean_diag.get('dualproxy_reliable_query', mean_diag.get('gtproxy_reliable_query', mean_diag.get('supervised_proto_positive', mean_diag.get('ft_proto_positive', mean_diag.get('prototype_positive', 0)))))))):.2f}, "
                f"rejected={mean_diag.get('prr_rejected_matched', mean_diag.get('pgrp_rejected_matched', mean_diag.get('dualproxy_rejected_query', mean_diag.get('gtproxy_rejected_query', mean_diag.get('supervised_proto_rejected_positive', mean_diag.get('ft_proto_rejected_positive', mean_diag.get('prototype_rejected_positive', 0))))))):.2f}, "
                f"hard/random={mean_diag.get('prr_hard_background', mean_diag.get('pgrp_hard_background', mean_diag.get('potp_hard_background', mean_diag.get('dualproxy_hard_background', mean_diag.get('gtproxy_hard_background', mean_diag.get('supervised_proto_hard_negative', mean_diag.get('ft_proto_hard_negative', mean_diag.get('prototype_hard_background', 0)))))))):.2f}/"
                f"{mean_diag.get('prr_random_background', mean_diag.get('pgrp_random_background', mean_diag.get('potp_random_background', mean_diag.get('dualproxy_random_background', mean_diag.get('supervised_proto_random_negative', mean_diag.get('ft_proto_random_negative', mean_diag.get('prototype_random_background', 0))))))):.2f}, "
                f"sim_pos={mean_diag.get('prr_proto_positive', mean_diag.get('pgrp_proto_positive', mean_diag.get('potp_positive_score', mean_diag.get('gtproxy_query_similarity', mean_diag.get('ft_proto_positive_similarity_mean', mean_diag.get('prototype_positive_margin_mean', float('nan'))))))):.6f}, "
                f"sim_neg={mean_diag.get('prr_proto_background', mean_diag.get('pgrp_proto_background', mean_diag.get('potp_background_score', mean_diag.get('gtproxy_background_similarity', mean_diag.get('ft_proto_negative_similarity_mean', mean_diag.get('prototype_background_margin_mean', float('nan'))))))):.6f}",
                flush=True,
            )
        if int(getattr(args, 'debug_max_train_batches', 0)) > 0:
            print(
                f'[Debug-train-limit] epoch={epoch}, processed_batches={processed_steps}',
                flush=True,
            )
    return reg_tl, cls_tl, proto_tl, mean_diag


class Evaluator:
    def __init__(self, data_loader_val):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader_val
        self.gds =[]
        for index, sample in enumerate(data_loader_val.dataset.data):
            data = sample
            files = data_loader_val.dataset.files[index]
            sample = data_loader_val.dataset.read_data(data, files)
            self.gds.append(tuple(sample.values())[1:])
        self.num_classes = args.num_classes

    @torch.no_grad()
    def predict_scores(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        # scores[:, -1] = scores[:, -1]*0.3
        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            return deduplicate_scores(points[reserved_index], scores[reserved_index], 15)
        else:
            return points[reserved_index], classes[reserved_index]

    def draw_roc(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            if i % args.world_size != rank:
                continue

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes, pd_scores = self.predict_scores(model, images, apply_deduplication=True)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_gd_points = self.gds[i][c]
                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, _ = binary_match(category_pd_points, category_gd_points,
                                                threshold_distance=effective_matching_dis)
                    cls_rn[c] += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                right_num, match_pred_index, match_gd_index = binary_match_predAndgd(pd_points, gd_points,
                                                                                     threshold_distance=effective_matching_dis)
                matched_pd_points = pd_points[match_pred_index]
                det_rn += right_num

        if args.distributed:
            dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

            dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        det_r = det_rn / (det_tn + eps)
        det_p = det_rn / (det_pn + eps)
        det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

        cls_r = cls_rn / (cls_tn + eps)
        cls_p = cls_rn / (cls_pn + eps)
        cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

        metrics = {'检测指标':[det_p.item(), det_r.item(), det_f1.item()],
                   '分类精度': cls_p.tolist(), '分类召回': cls_r.tolist(), '分类F1': cls_f1.tolist(),
                   '分类指标':[cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()]}
        return metrics

    @torch.no_grad()
    def predict(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            return deduplicate(points[reserved_index], scores[reserved_index], 15)
        else:
            return points[reserved_index], classes[reserved_index]

    def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        mse_sum = 0.0
        mae_sum = 0.0
        total_matches = 0

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            if i % args.world_size != rank:
                continue

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_gd_points = self.gds[i][c]
                category_pred_scores = pred_scores[pd_classes == c][:, 0]

                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, matched_pred, matched_gd = self.get_tp(category_pd_points, category_pred_scores, category_gd_points, thr=effective_matching_dis)
                    cls_rn[c] += right_num

                    if right_num > 0:
                        squared_errors = np.sum((matched_pred - matched_gd) ** 2, axis=1)
                        absolute_errors = np.sum(np.abs(matched_pred - matched_gd), axis=1)
                        mse_sum += np.sum(squared_errors)
                        mae_sum += np.sum(absolute_errors)
                        total_matches += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                pred_scores_global = np.sum(pred_scores[:, :-1], axis=1)
                right_num_global, _, _ = self.get_tp(pd_points, pred_scores_global, gd_points, thr=effective_matching_dis)
                det_rn += right_num_global

        # 分布式汇总
        if args.distributed:
            mse_sum_tensor = torch.tensor(mse_sum).cuda(rank)
            mae_sum_tensor = torch.tensor(mae_sum).cuda(rank)
            total_matches_tensor = torch.tensor(total_matches).cuda(rank)
            dist.all_reduce(mse_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mae_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_matches_tensor, op=dist.ReduceOp.SUM)
            mse_sum = mse_sum_tensor.item()
            mae_sum = mae_sum_tensor.item()
            total_matches = total_matches_tensor.item()

        # 计算MSE和MAE
        mse = mse_sum / (total_matches + eps)
        mae = mae_sum / (total_matches + eps)

        # 原指标计算
        det_r = det_rn / (det_tn + eps)
        det_p = det_rn / (det_pn + eps)
        det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

        cls_r = cls_rn / (cls_tn + eps)
        cls_p = cls_rn / (cls_pn + eps)
        cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

        metrics = {
            '检测指标':[det_p.item(), det_r.item(), det_f1.item()],
            '分类精度': cls_p.tolist(),
            '分类召回': cls_r.tolist(),
            '分类F1': cls_f1.tolist(),
            '分类指标':[cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()],
            'MSE': mse,
            'MAE': mae
        }
        return metrics
    
    def get_tp(self, pred_points, pred_scores, gd_points, thr=15):
        sorted_indices = np.argsort(-pred_scores)
        sorted_pred_points = pred_points[sorted_indices]

        unmatched = np.ones(len(gd_points), dtype=bool)
        dis = S.distance_matrix(sorted_pred_points, gd_points)

        matched_pred_indices = []
        matched_gd_indices =[]

        for i in range(len(sorted_pred_points)):
            if not np.any(unmatched):
                break
            min_index_in_unmatched = np.argmin(dis[i, unmatched])
            original_gd_index = np.where(unmatched)[0][min_index_in_unmatched]
            distance = dis[i, original_gd_index]
            if distance <= thr:
                matched_pred_indices.append(i)
                matched_gd_indices.append(original_gd_index)
                unmatched[original_gd_index] = False

        # Convert indices to original pred_points order
        original_pred_indices = sorted_indices[matched_pred_indices]
        matched_pred = pred_points[original_pred_indices]
        matched_gd = gd_points[matched_gd_indices]

        return len(matched_pred), matched_pred, matched_gd

    # 修改：完全重构了生成三种特定图像逻辑，并在 img_3 上标注 TP/FP/FN 数量
    def visual_analysis(self, model, output_dir='vis_results/normal'):
        from skimage import io
        import numpy as np

        model.eval()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = os.path.basename(self.data_loader.dataset.files[i])
            print(f'Visualizing --- {img_name}')

            images = images.cuda(non_blocking=True)

            # 修改点：使用 predict_scores 直接获取得分用于计算 TP/FP/FN
            pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)

            # 保持创建 db file 不变
            # from np2db import output2db
            # template_db = r'/root/autodl-tmp/p2p-wq/home/zhangwanqi/code/code_cell_p2p/slice.db'
            # db_save_root = "/root/autodl-tmp/p2p-wq/home/zhangwanqi/code/code_cell_p2p/employ"
            # table_name = "None"
            # save_folder = os.path.join(db_save_root, img_name)
            # label_name = {0: "印戒细胞"}
            # label_markgroup = {0: 520}

            # output2db(pd_points, pd_classes, template_db, save_folder, label_markgroup, table_name)

            image = self.data_loader.dataset.read_data(self.data_loader.dataset.data[i],
                                                       self.data_loader.dataset.files[i])['image'].copy()
            
            # 初始化三种状态的图像底板
            img_1 = image.copy()  # 1-真实标注：绿色点
            img_2 = image.copy()  # 2-真实标注（绿色圈）+模型预测（红点点）
            img_3 = image.copy()  # 3-真实标注（绿色圈）+模型预测对的（红点）+假阳（蓝色十字）+漏检（紫色十字）

            # 统一颜色定义 (RGB 格式, skimage.io.imsave 需要 RGB)
            color_gt_green = (0, 255, 0)
            color_pred_red = (255, 0, 0)
            color_fp_blue = (0, 0, 255)
            color_fn_purple = (255, 0, 255)
            
            # 画笔粗细定义
            circle_radius = 14
            circle_thick = 3     # 圆圈厚一点
            dot_radius = 4

            # 1. 绘制真实标注 (Image 1, 2, 3通用)
            for c in range(self.num_classes):
                for (x, y) in self.gds[i][c].astype(int):
                    # 1-真实标注：绿色点
                    cv.circle(img_1, (x, y), radius=dot_radius, color=color_gt_green, thickness=-1)
                    # 2 & 3: 真实标注：绿色圈
                    cv.circle(img_2, (x, y), radius=circle_radius, color=color_gt_green, thickness=circle_thick)
                    cv.circle(img_3, (x, y), radius=circle_radius, color=color_gt_green, thickness=circle_thick)

            # 2. 绘制模型预测 (Image 2: 模型预测红点点)
            for (x, y) in pd_points.astype(int):
                cv.circle(img_2, (x, y), radius=dot_radius, color=color_pred_red, thickness=-1)

            # 3. 匹配并分析绘制 TP, FP, FN (Image 3专属)
            def get_unmatched(all_pts, matched_pts):
                if len(matched_pts) == 0: return all_pts
                if len(all_pts) == 0: return all_pts
                diff = all_pts[:, np.newaxis, :] - matched_pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dists = np.min(dists, axis=1)
                return all_pts[min_dists > 1e-5]

            total_tp = 0
            total_fp = 0
            total_fn = 0

            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_pred_scores = pred_scores[pd_classes == c][:, 0]
                category_gd_points = self.gds[i][c]

                if len(category_pd_points) > 0 and len(category_gd_points) > 0:
                    # 使用 get_tp 找出被正确匹配的点
                    right_num, matched_pred, matched_gd = self.get_tp(category_pd_points, category_pred_scores, category_gd_points, thr=args.match_dis)
                else:
                    matched_pred = np.array([])
                    matched_gd = np.array([])

                tp_pts = matched_pred
                fp_pts = get_unmatched(category_pd_points, matched_pred)
                fn_pts = get_unmatched(category_gd_points, matched_gd)

                total_tp += len(tp_pts)
                total_fp += len(fp_pts)
                total_fn += len(fn_pts)

                # TP 模型预测对的: 红点
                for (x, y) in tp_pts.astype(int):
                    cv.circle(img_3, (x, y), radius=dot_radius, color=color_pred_red, thickness=-1)
                
                # FP 假阳: 蓝色十字
                for (x, y) in fp_pts.astype(int):
                    cv.drawMarker(img_3, (x, y), color=color_fp_blue, markerType=cv.MARKER_CROSS, markerSize=12, thickness=circle_thick)
                
                # FN 漏检: 紫色十字
                for (x, y) in fn_pts.astype(int):
                    cv.drawMarker(img_3, (x, y), color=color_fn_purple, markerType=cv.MARKER_CROSS, markerSize=12, thickness=circle_thick)

            # 在 img_3 上标注统计信息
            text = f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}"
            cv.putText(img_3, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # 保存三张图片并清理后缀避免文件堆叠后缀
            img_base = os.path.splitext(img_name)[0]
            io.imsave(f"{output_dir}/{img_base}_1_gt_dots.jpg", img_1, check_contrast=False)
            io.imsave(f"{output_dir}/{img_base}_2_gt_pred.jpg", img_2, check_contrast=False)
            io.imsave(f"{output_dir}/{img_base}_3_analysis.jpg", img_3, check_contrast=False)


    @torch.no_grad()
    def match_procedure(self, model):
        model.eval()
        matcher = build_matcher(args)
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = self.data_loader.dataset.files[i].split('.')[0]
            print(f'Processing --- ({img_name})')

            images = images.cuda(non_blocking=True)
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)

            targets = {'gt_nums': lengths,
                       'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                       'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

            indices = matcher(outputs, targets)
            np.save(f'./tool/result/{img_name}_indices',
                    [indices[0][0].cpu().numpy(), indices[0][1].cpu().numpy()])

            cls_scores = outputs['cls_logits'][0].softmax(-1).cpu().numpy()
            points = outputs['pnt_coords'][0].cpu().numpy()

            reg_attn = outputs['reg_attn'][0, ..., 0].cpu().numpy()
            cls_attn = outputs['cls_attn'][0, ..., 0].cpu().numpy()

            np.save(f'./tool/result/{img_name}_scores', cls_scores)
            np.save(f'./tool/result/{img_name}_points', points.astype(int))
            np.save(f'./tool/result/{img_name}_cls_attn', reg_attn)
            np.save(f'./tool/result/{img_name}_reg_attn', cls_attn)


def eval(ckpt_path):
    model = build_model(args)
    ckpt = torch.load(f'checkpoint/{ckpt_path}', map_location='cpu')
    print(ckpt['epoch'], ckpt['metrics'])
    # ckpt = torch.load(f'./{ckpt_path}', map_location='cpu')

    dataset_val = build_dataset(args, 'test')
    # dataset_val = build_dataset(args, 'train')
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn_pad)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()

    evaluator = Evaluator(data_loader_val)
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)
    
    print(metrics)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # os.environ['MASTER_PORT'] = '29600'

    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    train()
    # test()
