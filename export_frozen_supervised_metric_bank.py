import argparse
import csv
import hashlib
import json
import os
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch

from diagnose_class_conditional_prototypes import (
    TYPE_HARD_NEGATIVE,
    TYPE_POSITIVE,
    TYPE_RANDOM_NEGATIVE,
)
from diagnose_supervised_metric_prototypes import (
    _macro_group_auc,
    fit_supervised_prototypes,
    make_gate_decision,
    score_supervised_prototypes,
    subset_by_group_and_class,
)
from models.frozen_supervised_metric_prototype import (
    FROZEN_SUPERVISED_METRIC_VERSION,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and freeze the source-supervised three-class prototype bank."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--embedding_dim", default=32, type=int)
    parser.add_argument("--foreground_prototypes", default=4, type=int)
    parser.add_argument("--hard_background_prototypes", default=4, type=int)
    parser.add_argument("--random_background_prototypes", default=2, type=int)
    parser.add_argument("--temperature", default=0.15, type=float)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--calibration_fraction", default=0.20, type=float)
    parser.add_argument("--positive_per_group", default=4096, type=int)
    parser.add_argument("--hard_negative_per_group", default=2048, type=int)
    parser.add_argument("--random_negative_per_group", default=2048, type=int)
    parser.add_argument("--hard_negative_weight", default=2.0, type=float)
    parser.add_argument("--diversity_weight", default=0.02, type=float)
    parser.add_argument("--projection_anchor_weight", default=1e-3, type=float)
    parser.add_argument("--kmeans_iterations", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hard_auc_gate", default=0.63, type=float)
    parser.add_argument("--hard_min_auc_gate", default=0.55, type=float)
    parser.add_argument("--random_auc_gate", default=0.95, type=float)
    parser.add_argument("--teacher_gain_gate", default=0.05, type=float)
    return parser


def build_groupwise_train_calibration_split(
    group_index: np.ndarray,
    image_index: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.zeros(len(group_index), dtype=bool)
    calibration_mask = np.zeros(len(group_index), dtype=bool)
    for group in sorted(int(value) for value in np.unique(group_index)):
        group_mask = group_index == group
        images = np.unique(image_index[group_mask])
        if len(images) < 2:
            raise RuntimeError(f"group={group} needs at least two images")
        count = max(1, int(round(len(images) * calibration_fraction)))
        count = min(count, len(images) - 1)
        random = np.random.RandomState(seed + 1009 * group)
        calibration_images = random.choice(images, size=count, replace=False)
        selected_calibration = group_mask & np.isin(image_index, calibration_images)
        calibration_mask |= selected_calibration
        train_mask |= group_mask & ~selected_calibration
    train = np.where(train_mask)[0]
    calibration = np.where(calibration_mask)[0]
    if set(image_index[train]).intersection(set(image_index[calibration])):
        raise RuntimeError("full-source train/calibration image leakage")
    return train, calibration


def _sha256_state(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_history(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.features) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    required = {
        "features",
        "candidate_type",
        "teacher_score",
        "group_index",
        "image_index",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise RuntimeError(f"candidate archive missing arrays: {missing}")
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("feature_source") != "p2p_cls_features_before_final_classifier":
        raise RuntimeError("candidate archive feature source is incompatible")
    if not manifest.get("checkpoint_sha256"):
        raise RuntimeError("candidate manifest lacks teacher checkpoint SHA256")

    features = data["features"].astype(np.float32, copy=False)
    candidate_type = data["candidate_type"].astype(np.int64, copy=False)
    group_index = data["group_index"].astype(np.int64, copy=False)
    image_index = data["image_index"].astype(np.int64, copy=False)
    teacher_score = data["teacher_score"].astype(np.float32, copy=False)
    train_all, calibration_all = build_groupwise_train_calibration_split(
        group_index,
        image_index,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    caps = (
        args.positive_per_group,
        args.hard_negative_per_group,
        args.random_negative_per_group,
    )
    train_indices = subset_by_group_and_class(
        train_all, candidate_type, group_index, caps, args.seed + 200000
    )
    calibration_indices = subset_by_group_and_class(
        calibration_all, candidate_type, group_index, caps, args.seed + 300000
    )
    fit_args = SimpleNamespace(**vars(args))
    model, history, model_audit = fit_supervised_prototypes(
        features,
        candidate_type,
        train_indices,
        calibration_indices,
        fit_args,
        device,
        group_index=group_index,
    )
    scores, assignments = score_supervised_prototypes(
        model, features[calibration_indices], args.batch_size
    )
    local_types = candidate_type[calibration_indices]
    local_groups = group_index[calibration_indices]
    hard_auc, hard_min_auc = _macro_group_auc(
        scores, local_types, local_groups, TYPE_POSITIVE, TYPE_HARD_NEGATIVE
    )
    random_auc, _ = _macro_group_auc(
        scores, local_types, local_groups, TYPE_POSITIVE, TYPE_RANDOM_NEGATIVE
    )
    teacher_hard_auc, _ = _macro_group_auc(
        teacher_score[calibration_indices],
        local_types,
        local_groups,
        TYPE_POSITIVE,
        TYPE_HARD_NEGATIVE,
    )
    decision = make_gate_decision(
        hard_auc=hard_auc,
        hard_min_auc=hard_min_auc,
        random_auc=random_auc,
        teacher_hard_auc=teacher_hard_auc,
        ridge_hard_auc=hard_auc,
        hard_auc_gate=args.hard_auc_gate,
        hard_min_auc_gate=args.hard_min_auc_gate,
        random_auc_gate=args.random_auc_gate,
        teacher_gain_gate=args.teacher_gain_gate,
    )
    model_hash = _sha256_state(model)
    bank_id = f"supervised-metric-{model_hash[:16]}"
    metadata = {
        "version": FROZEN_SUPERVISED_METRIC_VERSION,
        "gate_pass": bool(decision["gate_pass"]),
        "feature_source": manifest["feature_source"],
        "checkpoint": manifest.get("checkpoint", ""),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_epoch": manifest.get("checkpoint_epoch"),
        "candidate_archive": os.path.abspath(args.features),
        "manifest": os.path.abspath(args.manifest),
        "train_images": int(len(np.unique(image_index[train_indices]))),
        "calibration_images": int(len(np.unique(image_index[calibration_indices]))),
        "train_candidates": int(len(train_indices)),
        "calibration_candidates": int(len(calibration_indices)),
        "selected_epoch": int(model_audit["selected_epoch"]),
        "configuration": vars(args),
        "decision": decision,
    }
    payload = {
        "projector_weight": model.projector.weight.detach().cpu(),
        "projector_bias": model.projector.bias.detach().cpu(),
        "prototypes": model.normalized_prototypes().detach().cpu(),
        "prototype_counts": tuple(model.prototype_counts),
        "temperature": float(model.temperature),
        "bank_id": bank_id,
        "metadata": metadata,
    }
    audit = {
        "bank_id": bank_id,
        "model_state_sha256": model_hash,
        "gate": decision,
        "model": model_audit,
        "candidate_counts": {
            "train": {
                "positive": int((candidate_type[train_indices] == TYPE_POSITIVE).sum()),
                "hard_negative": int(
                    (candidate_type[train_indices] == TYPE_HARD_NEGATIVE).sum()
                ),
                "random_negative": int(
                    (candidate_type[train_indices] == TYPE_RANDOM_NEGATIVE).sum()
                ),
            },
            "calibration": {
                "positive": int(
                    (candidate_type[calibration_indices] == TYPE_POSITIVE).sum()
                ),
                "hard_negative": int(
                    (candidate_type[calibration_indices] == TYPE_HARD_NEGATIVE).sum()
                ),
                "random_negative": int(
                    (candidate_type[calibration_indices] == TYPE_RANDOM_NEGATIVE).sum()
                ),
            },
        },
        "assignment_counts": np.bincount(
            assignments, minlength=sum(model.prototype_counts)
        ).astype(int).tolist(),
    }
    with open(
        os.path.join(args.output_dir, "frozen_supervised_metric_bank_audit.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    _write_history(
        os.path.join(args.output_dir, "training_history.csv"), history
    )
    if not decision["gate_pass"]:
        raise RuntimeError(
            "full-source frozen supervised prototype gate failed; training is blocked"
        )
    output_path = os.path.join(
        args.output_dir, "frozen_supervised_metric_bank.pth"
    )
    temporary = output_path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    print(
        f"[Frozen-supervised-bank] gate_pass=True, hard_auc={hard_auc:.6f}, "
        f"hard_min={hard_min_auc:.6f}, random_auc={random_auc:.6f}",
        flush=True,
    )
    print(f"[Frozen-supervised-bank] output={output_path}", flush=True)


if __name__ == "__main__":
    main()
