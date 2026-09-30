import json
import copy
import itertools
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import collate_fn_pad

import train_p2p as base
from transforms import Preprocessing


KEY_DET = "\u68c0\u6d4b\u6307\u6807"
KEY_CLS_PRECISION = "\u5206\u7c7b\u7cbe\u5ea6"
KEY_CLS_RECALL = "\u5206\u7c7b\u53ec\u56de"
KEY_CLS_F1 = "\u5206\u7c7bF1"
KEY_CLS_METRICS = "\u5206\u7c7b\u6307\u6807"


def is_main():
    return getattr(base.args, "rank", 0) == 0


def has_positive_annotation(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("annotation", [])) > 0


def write_list(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row) + "\n")


def filter_non_empty_dataset(dataset, phase_name):
    original_size = len(dataset.data)
    kept_data, kept_files, skipped_files = [], [], []
    for json_path, image_path in zip(dataset.data, dataset.files):
        if has_positive_annotation(json_path):
            kept_data.append(json_path)
            kept_files.append(image_path)
        else:
            skipped_files.append(image_path)

    dataset.data = kept_data
    dataset.files = kept_files

    if is_main():
        print(
            f"[NoEmpty-{phase_name}] original={original_size}, kept={len(kept_files)}, "
            f"skipped_empty={len(skipped_files)}",
            flush=True,
        )
        if getattr(base.args, "output_dir", ""):
            write_list(os.path.join(base.args.output_dir, f"{phase_name}_non_empty_files.txt"), kept_files)
            write_list(os.path.join(base.args.output_dir, f"{phase_name}_skipped_empty_files.txt"), skipped_files)

    if not kept_files:
        raise RuntimeError(f"No non-empty samples left after filtering {phase_name} dataset.")
    return dataset


def assert_dataset_phase(args, image_set, dataset):
    expected_phase = getattr(args, "eval_split", "test") if image_set == "test" else image_set
    actual_phase = getattr(dataset, "phase", None)
    if actual_phase != expected_phase:
        raise RuntimeError(
            "Dataset phase mismatch. "
            f"image_set={image_set}, expected_phase={expected_phase}, actual_phase={actual_phase}. "
            "This usually means dataset_zy_src.py is outdated and does not honor --eval_split. "
            "Training is stopped to avoid using test split as validation."
        )
    return expected_phase


def build_dataset_no_empty(args, image_set):
    dataset = base._original_build_dataset(args, image_set)
    phase_name = assert_dataset_phase(args, image_set, dataset)
    if image_set == "train" and getattr(args, "filter_empty_train", True):
        return filter_non_empty_dataset(dataset, phase_name)
    if image_set == "test" and getattr(args, "skip_empty_eval", True):
        return filter_non_empty_dataset(dataset, phase_name)
    return dataset


def _slide_group(image_path):
    name = os.path.basename(image_path)
    lower = name.lower()
    marker = lower.find('.kfb_')
    if marker >= 0:
        return name[:marker + 4]
    return os.path.splitext(name)[0]


def _choose_validation_groups(group_rows, target_count):
    groups = sorted(group_rows)
    if len(groups) < 2:
        raise RuntimeError('E2 validation requires at least two source slide groups')
    if len(groups) > 20:
        raise RuntimeError(
            f'E2 slide holdout supports at most 20 groups, got {len(groups)}; '
            'create an explicit validation split instead'
        )
    candidates = []
    for count in range(1, len(groups)):
        for subset in itertools.combinations(groups, count):
            rows = sum(len(group_rows[group]) for group in subset)
            candidates.append((abs(rows - target_count), count, subset))
    return min(candidates)[2]


def construct_dataset_e2():
    train_dataset = build_dataset_no_empty(base.args, 'train')
    ordered = sorted(zip(train_dataset.files, train_dataset.data))
    train_dataset.files = [row[0] for row in ordered]
    train_dataset.data = [row[1] for row in ordered]
    groups = {}
    for index, image_path in enumerate(train_dataset.files):
        groups.setdefault(_slide_group(image_path), []).append(index)
    target_count = max(1, int(round(len(train_dataset.files) * base.args.e2_val_fraction)))
    val_groups = set(_choose_validation_groups(groups, target_count))
    val_indices = [i for group in sorted(val_groups) for i in groups[group]]
    train_indices = [i for i, path in enumerate(train_dataset.files)
                     if _slide_group(path) not in val_groups]
    if not train_indices or not val_indices:
        raise RuntimeError('E2 slide holdout produced an empty train or validation split')

    val_dataset = copy.copy(train_dataset)
    val_dataset.data = [train_dataset.data[i] for i in val_indices]
    val_dataset.files = [train_dataset.files[i] for i in val_indices]
    val_dataset.phase = 'val'
    mean, std = np.load(train_dataset.mean_std_path)
    val_dataset.data_transform = Preprocessing(mean, std)
    train_dataset.data = [train_dataset.data[i] for i in train_indices]
    train_dataset.files = [train_dataset.files[i] for i in train_indices]

    if is_main() and base.args.output_dir:
        split = {
            'protocol': 'deterministic_slide_group_holdout_from_train',
            'seed': int(base.args.seed),
            'target_val_fraction': float(base.args.e2_val_fraction),
            'actual_val_fraction': len(val_indices) / len(ordered),
            'val_groups': sorted(val_groups),
            'train_groups': sorted(set(groups) - val_groups),
            'train_files': train_dataset.files,
            'val_files': val_dataset.files,
            'caveat': 'The fixed baseline checkpoint may have been trained on these held-out source slides; this validates E2 selection, not an independent P2P baseline.'
        }
        path = os.path.join(base.args.output_dir, 'e2_validation_split.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(split, handle, ensure_ascii=False, indent=2)
        print(
            f'[E2-split] train={len(train_indices)}, val={len(val_indices)}, '
            f'val_groups={sorted(val_groups)}, manifest={path}', flush=True
        )

    if base.args.distributed:
        sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, sampler=sampler, batch_size=base.args.batch_size,
            num_workers=0, collate_fn=collate_fn_pad,
        )
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=base.args.batch_size,
            num_workers=0, collate_fn=collate_fn_pad,
        )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=0,
        collate_fn=collate_fn_pad,
    )
    return {'train': train_loader, 'test': val_loader}


def calculate_metrics_skip_empty(self, model, effective_matching_dis=12, rank=0):
    eps = 1e-8
    model.eval()

    cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
    det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
    mse_sum = 0.0
    mae_sum = 0.0
    total_matches = 0.0
    total_eval_images = 0.0
    metric_images = 0.0
    skipped_empty_images = 0.0
    skipped_files = []

    for i, (images, points, labels, lengths) in enumerate(self.data_loader):
        if i % base.args.world_size != rank:
            continue

        total_eval_images += 1
        images = images.cuda(non_blocking=True)
        pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)
        gd_points = np.zeros((0, 2), dtype=int)
        per_class_items = []

        for c in range(self.num_classes):
            category_pd_points = pd_points[pd_classes == c]
            category_gd_points = self.gds[i][c]
            category_pred_scores = pred_scores[pd_classes == c][:, 0]
            gd_points = np.concatenate([gd_points, category_gd_points], axis=0)
            per_class_items.append((c, category_pd_points, category_gd_points, category_pred_scores))

        if len(gd_points) == 0:
            skipped_empty_images += 1
            if hasattr(self.data_loader.dataset, "files"):
                skipped_files.append(self.data_loader.dataset.files[i])
            continue

        metric_images += 1
        for c, category_pd_points, category_gd_points, category_pred_scores in per_class_items:
            pred_num, gd_num = len(category_pd_points), len(category_gd_points)
            cls_pn[c] += pred_num
            cls_tn[c] += gd_num

            if pred_num and gd_num:
                right_num, matched_pred, matched_gd = self.get_tp(
                    category_pd_points, category_pred_scores, category_gd_points,
                    thr=effective_matching_dis)
                cls_rn[c] += right_num

                if right_num > 0:
                    mse_sum += float(np.sum((matched_pred - matched_gd) ** 2))
                    mae_sum += float(np.sum(np.abs(matched_pred - matched_gd)))
                    total_matches += float(right_num)

        det_pn += len(pd_points)
        det_tn += len(gd_points)
        if len(pd_points) and len(gd_points):
            pred_scores_global = np.sum(pred_scores[:, :-1], axis=1)
            right_num_global, _, _ = self.get_tp(
                pd_points, pred_scores_global, gd_points, thr=effective_matching_dis)
            det_rn += right_num_global

    if base.args.distributed:
        mse_sum_tensor = torch.tensor(float(mse_sum), dtype=torch.float64).cuda(rank)
        mae_sum_tensor = torch.tensor(float(mae_sum), dtype=torch.float64).cuda(rank)
        total_matches_tensor = torch.tensor(float(total_matches), dtype=torch.float64).cuda(rank)
        eval_count_tensor = torch.tensor(
            [total_eval_images, metric_images, skipped_empty_images],
            dtype=torch.float64
        ).cuda(rank)
        for tensor in (det_rn, det_tn, det_pn, cls_pn, cls_tn, cls_rn):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        for tensor in (mse_sum_tensor, mae_sum_tensor, total_matches_tensor, eval_count_tensor):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        mse_sum = mse_sum_tensor.item()
        mae_sum = mae_sum_tensor.item()
        total_matches = total_matches_tensor.item()
        total_eval_images, metric_images, skipped_empty_images = eval_count_tensor.cpu().tolist()

    mse = mse_sum / (total_matches + eps)
    mae = mae_sum / (total_matches + eps)
    det_r = det_rn / (det_tn + eps)
    det_p = det_rn / (det_pn + eps)
    det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)
    cls_r = cls_rn / (cls_tn + eps)
    cls_p = cls_rn / (cls_pn + eps)
    cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

    metrics = {
        KEY_DET: [det_p.item(), det_r.item(), det_f1.item()],
        KEY_CLS_PRECISION: cls_p.tolist(),
        KEY_CLS_RECALL: cls_r.tolist(),
        KEY_CLS_F1: cls_f1.tolist(),
        KEY_CLS_METRICS: [cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()],
        "MSE": mse,
        "MAE": mae,
        "eval_protocol": "skip_empty_gt",
        "eval_counts": {
            "metric_images": float(metric_images),
            "total_eval_images": float(total_eval_images),
            "skipped_empty_images": float(skipped_empty_images),
            "det_tp": det_rn.item(),
            "det_pred": det_pn.item(),
            "det_gt": det_tn.item(),
            "matched_points": float(total_matches),
        },
    }

    if is_main():
        print(
            "[NoEmpty-eval] "
            f"metric_images={metric_images:.0f}, total_eval_images={total_eval_images:.0f}, "
            f"skipped_empty={skipped_empty_images:.0f}, "
            f"P={det_p.item():.6f}, R={det_r.item():.6f}, F1={det_f1.item():.6f}",
            flush=True,
        )
        if getattr(base.args, "output_dir", "") and skipped_files:
            write_list(os.path.join(base.args.output_dir, "eval_skipped_empty_files_rank0.txt"), skipped_files)
    return metrics


def add_mean_std_args(parser):
    parser.add_argument('--mean_std_path', default='', type=str,
                        help='optional mean_std.npy path shared by train/test; default uses ./datasets/{dataset}/mean_std.npy')
    parser.add_argument('--train_mean_std_path', default='', type=str,
                        help='optional train split mean_std.npy path; overrides --mean_std_path for train')
    parser.add_argument('--test_mean_std_path', default='', type=str,
                        help='optional test split mean_std.npy path; overrides --mean_std_path for test')


def get_args_parser():
    parser = base.get_args_parser()
    add_mean_std_args(parser)
    parser.add_argument("--eval_split", default="test", choices=("val", "test"),
                        help="split directory used when train_p2p.py requests the test/eval dataset")
    parser.add_argument("--filter_empty_train", action="store_true", default=True,
                        help="filter empty-GT patches from train split")
    parser.add_argument("--keep_empty_train", dest="filter_empty_train", action="store_false",
                        help="keep empty-GT train patches")
    parser.add_argument("--skip_empty_eval", action="store_true", default=True,
                        help="skip empty-GT patches from eval metrics")
    parser.add_argument("--include_empty_eval", dest="skip_empty_eval", action="store_false",
                        help="include empty-GT patches in eval metrics")
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    base.args = args
    base._original_build_dataset = base.build_dataset
    base.build_dataset = build_dataset_no_empty
    if args.use_proto_e2:
        base.construct_dataset = construct_dataset_e2
    if args.skip_empty_eval:
        base.Evaluator.calculate_metrics = calculate_metrics_skip_empty

    base.init_distributed_mode(args)
    seed = args.seed + base.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if is_main():
        print("[NoEmptyTrainEntry] entry=train_p2p_no_empty_v2.py, base=train_p2p.py", flush=True)
        print("[NoEmptyTrainEntry] default_protocol=filter_empty_train + filter_empty_test + skip_empty_eval_safety", flush=True)
        print(
            f"[NoEmptyTrainEntry] filter_empty_train={args.filter_empty_train}, "
            f"skip_empty_eval={args.skip_empty_eval}, dataset={args.dataset}, "
            f"eval_split={args.eval_split}, output_dir={args.output_dir}",
            flush=True,
        )
        print(
            f"[NoEmptyTrainEntry] mean_std_path={args.mean_std_path or '<dataset_default>'}, "
            f"train_mean_std_path={args.train_mean_std_path or '<unset>'}, "
            f"test_mean_std_path={args.test_mean_std_path or '<unset>'}",
            flush=True,
        )

    base.train()


if __name__ == "__main__":
    main()
