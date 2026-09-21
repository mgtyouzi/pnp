import argparse
import csv
import hashlib
import json
import math
import os
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from build_frozen_teacher_fg_bank import spherical_kmeans
from candidate_conditioned_proto_data import (
    CATEGORY_BACKGROUND_FAR_FP,
    CATEGORY_ID_TO_NAME,
    CATEGORY_NEAR_MISS_FP,
    CATEGORY_TP,
    build_candidate_feature_matrix,
    build_loso_image_split,
    candidate_conditioned_gate,
    group_balanced_sample_weights,
    score_matched_group_sample,
)
from frozen_proto_rescoring import _binary_roc_auc
from models.candidate_conditioned_prototype import (
    CANDIDATE_CONDITIONED_PROTO_VERSION,
    CandidateConditionedPrototype,
)
from spatial_morphology_proto_data import validate_spatial_archive_contract


AUDIT_VERSION = "candidate_conditioned_proto_loso_v1_20260828"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free, slide-balanced TP-vs-background-far prototype audit."
        )
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--feature_mode",
        default="cls_reg_context",
        choices=("cls_only", "cls_reg_context", "spatial_morphology"),
    )
    parser.add_argument("--embedding_dim", default=32, type=int)
    parser.add_argument("--foreground_prototypes", default=4, type=int)
    parser.add_argument("--background_prototypes", default=4, type=int)
    parser.add_argument("--temperature", default=0.15, type=float)
    parser.add_argument("--learning_rate", default=0.005, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--calibration_fraction", default=0.20, type=float)
    parser.add_argument("--score_bins", default=5, type=int)
    parser.add_argument("--max_per_class_per_bin", default=512, type=int)
    parser.add_argument("--diversity_weight", default=0.02, type=float)
    parser.add_argument("--projection_anchor_weight", default=1e-3, type=float)
    parser.add_argument("--kmeans_iterations", default=20, type=int)
    parser.add_argument("--beta_grid", default="0,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--conditional_score_bins", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--require_gate", action="store_true")
    return parser


def _write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_grid(value: str) -> List[float]:
    values = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or values[0] < 0 or values[-1] > 1:
        raise ValueError("beta_grid must contain values in [0, 1]")
    return values


def _macro_auc(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray
) -> Tuple[float, float, Dict[int, float]]:
    values = {}
    for group in sorted(int(value) for value in np.unique(groups)):
        local = groups == group
        auc = _binary_roc_auc(labels[local], scores[local])
        if np.isfinite(auc):
            values[group] = float(auc)
    if not values:
        return float("nan"), float("nan"), {}
    aucs = list(values.values())
    return float(np.mean(aucs)), float(np.min(aucs)), values


def _pair_indices(categories: np.ndarray, indices: np.ndarray, negative: int):
    indices = np.asarray(indices, dtype=np.int64)
    return indices[np.isin(categories[indices], [CATEGORY_TP, negative])]


def _labels(categories: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return (categories[indices] == CATEGORY_TP).astype(np.int64)


def _initialize_projection(
    model: CandidateConditionedPrototype,
    features: torch.Tensor,
    labels: torch.Tensor,
    kmeans_iterations: int,
    seed: int,
    ridge: float = 0.01,
) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1, eps=1e-6)
    x = torch.cat(
        [normalized, torch.ones((len(normalized), 1), device=features.device)], dim=1
    )
    target = F.one_hot(labels.long(), num_classes=2).float().mul(2.0).sub(1.0)
    identity = torch.eye(x.shape[1], device=features.device)
    identity[-1, -1] = 0.0
    solution = torch.linalg.solve(
        x.t().matmul(x) + ridge * identity,
        x.t().matmul(target),
    )
    with torch.no_grad():
        dimensions = min(2, model.embedding_dim)
        model.projector.weight[:dimensions].copy_(
            solution[:-1, :dimensions].t()
        )
        model.projector.bias[:dimensions].copy_(solution[-1, :dimensions])
        embedded = model.embed(features)
        centers = []
        for class_id, count in enumerate(model.prototype_counts):
            local = embedded[labels == class_id]
            prototype, _ = spherical_kmeans(
                local,
                count,
                iterations=kmeans_iterations,
                seed=seed + 1009 * class_id,
            )
            centers.append(prototype)
        model.prototypes.copy_(torch.cat(centers, dim=0))
    return model.projector.weight.detach().clone()


def _diversity_loss(model: CandidateConditionedPrototype) -> torch.Tensor:
    prototypes = model.normalized_prototypes()
    penalties = []
    for start, end in model.class_slices:
        count = end - start
        if count <= 1:
            continue
        similarities = prototypes[start:end].matmul(prototypes[start:end].t())
        off_diagonal = ~torch.eye(
            count, dtype=torch.bool, device=prototypes.device
        )
        penalties.append(F.relu(similarities[off_diagonal] - 0.50).mean())
    return torch.stack(penalties).mean() if penalties else prototypes.sum() * 0.0


@torch.no_grad()
def _score_model(
    model: CandidateConditionedPrototype,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    scores = []
    model.eval()
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start : start + batch_size]).float().to(device)
        scores.append(model.margin(values).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


def _fit_model(
    features: np.ndarray,
    categories: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    args,
    device: torch.device,
    fixed_epochs: int = 0,
):
    if not len(train_indices):
        raise RuntimeError("score-matched prototype training set is empty")
    train_labels_np = (categories[train_indices] == CATEGORY_BACKGROUND_FAR_FP).astype(
        np.int64
    )
    if set(train_labels_np.tolist()) != {0, 1}:
        raise RuntimeError("prototype training needs both TP and background-far FP")
    model = CandidateConditionedPrototype(
        feat_dim=features.shape[1],
        embedding_dim=args.embedding_dim,
        prototype_counts=(args.foreground_prototypes, args.background_prototypes),
        temperature=args.temperature,
    ).to(device)
    train_values = torch.from_numpy(features[train_indices]).float().to(device)
    train_labels = torch.from_numpy(train_labels_np).long().to(device)
    train_weights = torch.from_numpy(
        group_balanced_sample_weights(groups, train_indices)
    ).float().to(device)
    anchor_weight = _initialize_projection(
        model,
        train_values,
        train_labels,
        kmeans_iterations=args.kmeans_iterations,
        seed=args.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    history = []
    best_key = (-float("inf"), -float("inf"), 0)
    best_state = None
    stale = 0
    maximum_epochs = fixed_epochs or args.epochs
    for epoch in range(maximum_epochs):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            values = train_values[batch]
            labels = train_labels[batch]
            weights = train_weights[batch]
            logits = model.class_logits(values)
            per_candidate = F.cross_entropy(logits, labels, reduction="none")
            classification = (per_candidate * weights).sum() / weights.sum().clamp_min(
                1e-8
            )
            diversity = _diversity_loss(model)
            anchor = F.mse_loss(model.projector.weight, anchor_weight)
            loss = (
                classification
                + args.diversity_weight * diversity
                + args.projection_anchor_weight * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        if fixed_epochs:
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
            continue

        calibration = _pair_indices(
            categories, calibration_indices, CATEGORY_BACKGROUND_FAR_FP
        )
        scores = _score_model(model, features[calibration], args.batch_size)
        labels = _labels(categories, calibration)
        macro, minimum, _ = _macro_auc(labels, scores, groups[calibration])
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "calibration_macro_auc": macro,
            "calibration_min_group_auc": minimum,
        }
        history.append(row)
        key = (macro, minimum, -(epoch + 1))
        if key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if not fixed_epochs:
        if best_state is None:
            raise RuntimeError("prototype calibration never produced a finite model")
        model.load_state_dict(best_state)
    return model, history


def _fit_joint_calibration(
    raw: np.ndarray,
    prototype: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    beta_grid: Iterable[float],
) -> Dict[str, float]:
    raw_mean, raw_std = float(raw.mean()), float(raw.std())
    proto_mean, proto_std = float(prototype.mean()), float(prototype.std())
    raw_std = max(raw_std, 1e-8)
    proto_std = max(proto_std, 1e-8)
    raw_z = (raw - raw_mean) / raw_std
    proto_z = (prototype - proto_mean) / proto_std
    best_key = (-float("inf"), -float("inf"), 0.0)
    best_beta = 0.0
    for beta in beta_grid:
        score = (1.0 - beta) * raw_z + beta * proto_z
        macro, minimum, _ = _macro_auc(labels, score, groups)
        key = (macro, minimum, -float(beta))
        if key > best_key:
            best_key = key
            best_beta = float(beta)
    return {
        "beta": best_beta,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "prototype_mean": proto_mean,
        "prototype_std": proto_std,
    }


def _joint_score(
    raw: np.ndarray, prototype: np.ndarray, calibration: Dict[str, float]
) -> np.ndarray:
    raw_z = (raw - calibration["raw_mean"]) / calibration["raw_std"]
    proto_z = (
        prototype - calibration["prototype_mean"]
    ) / calibration["prototype_std"]
    beta = calibration["beta"]
    return (1.0 - beta) * raw_z + beta * proto_z


def _conditional_auc(
    labels: np.ndarray,
    raw: np.ndarray,
    prototype: np.ndarray,
    groups: np.ndarray,
    bins: int,
) -> float:
    values = []
    for group in np.unique(groups):
        local = np.flatnonzero(groups == group)
        edges = np.unique(np.quantile(raw[local], np.linspace(0.0, 1.0, bins + 1)))
        local_bins = np.digitize(
            raw[local], edges[1:-1] if len(edges) > 2 else [], right=False
        )
        for bin_index in np.unique(local_bins):
            selected = local[local_bins == bin_index]
            auc = _binary_roc_auc(labels[selected], prototype[selected])
            if np.isfinite(auc):
                values.append(float(auc))
    return float(np.mean(values)) if values else float("nan")


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
    with np.load(args.archive) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"category", "raw_margin", "group_index", "image_index"}
    required.add(
        "morphology_features"
        if args.feature_mode == "spatial_morphology"
        else "features"
    )
    missing = sorted(required.difference(data))
    if missing:
        raise RuntimeError(f"candidate archive missing arrays: {missing}")
    if manifest.get("phase") != "train":
        raise RuntimeError("candidate-conditioned bank must be fitted on source train")
    if args.feature_mode == "spatial_morphology":
        validate_spatial_archive_contract(manifest)

    features, feature_source = build_candidate_feature_matrix(data, args.feature_mode)
    available_sources = dict(manifest.get("available_feature_sources", {}))
    if available_sources.get(args.feature_mode) != feature_source:
        raise RuntimeError("candidate archive feature contract is incompatible")
    categories = data["category"].astype(np.int64, copy=False)
    raw_margin = data["raw_margin"].astype(np.float64, copy=False)
    groups = data["group_index"].astype(np.int64, copy=False)
    images = data["image_index"].astype(np.int64, copy=False)
    eligible = np.flatnonzero(
        np.isin(categories, [CATEGORY_TP, CATEGORY_BACKGROUND_FAR_FP])
    )
    if len(np.unique(groups[eligible])) < 3:
        raise RuntimeError("candidate-conditioned LOSO needs at least three slide groups")

    fold_rows = []
    candidate_rows = []
    selected_epochs = []
    score_match_audit = []
    beta_grid = _parse_grid(args.beta_grid)
    for held_group in sorted(int(value) for value in np.unique(groups[eligible])):
        split = build_loso_image_split(
            groups,
            images,
            held_group=held_group,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
        )
        train_selected, train_audit = score_matched_group_sample(
            split.train_indices,
            categories,
            groups,
            raw_margin,
            score_bins=args.score_bins,
            max_per_class_per_bin=args.max_per_class_per_bin,
            seed=args.seed + 100003 * held_group,
        )
        for row in train_audit:
            score_match_audit.append({"held_group": held_group, **row})
        calibration_indices = _pair_indices(
            categories, split.calibration_indices, CATEGORY_BACKGROUND_FAR_FP
        )
        model, history = _fit_model(
            features,
            categories,
            groups,
            train_selected,
            calibration_indices,
            args,
            device,
        )
        _write_csv(
            os.path.join(args.output_dir, f"fold_{held_group}_history.csv"), history
        )
        selected_epoch = int(
            max(
                history,
                key=lambda row: (
                    row.get("calibration_macro_auc", -np.inf),
                    row.get("calibration_min_group_auc", -np.inf),
                    -row["epoch"],
                ),
            )["epoch"]
        )
        selected_epochs.append(selected_epoch)

        calibration_proto = _score_model(
            model, features[calibration_indices], args.batch_size
        )
        calibration = _fit_joint_calibration(
            raw_margin[calibration_indices],
            calibration_proto,
            _labels(categories, calibration_indices),
            groups[calibration_indices],
            beta_grid,
        )
        held_indices = split.test_indices
        held_proto = _score_model(model, features[held_indices], args.batch_size)
        held_joint = _joint_score(
            raw_margin[held_indices], held_proto, calibration
        )
        for local, archive_index in enumerate(held_indices):
            candidate_rows.append(
                {
                    "held_group": held_group,
                    "archive_index": int(archive_index),
                    "group_index": int(groups[archive_index]),
                    "image_index": int(images[archive_index]),
                    "category": CATEGORY_ID_TO_NAME.get(
                        int(categories[archive_index]), "unknown"
                    ),
                    "raw_margin": float(raw_margin[archive_index]),
                    "prototype_margin": float(held_proto[local]),
                    "joint_score": float(held_joint[local]),
                }
            )
        pair_local = np.isin(
            categories[held_indices], [CATEGORY_TP, CATEGORY_BACKGROUND_FAR_FP]
        )
        pair_indices = held_indices[pair_local]
        labels = _labels(categories, pair_indices)
        raw_auc = _binary_roc_auc(labels, raw_margin[pair_indices])
        proto_auc = _binary_roc_auc(labels, held_proto[pair_local])
        joint_auc = _binary_roc_auc(labels, held_joint[pair_local])
        near_local = np.isin(
            categories[held_indices], [CATEGORY_TP, CATEGORY_NEAR_MISS_FP]
        )
        near_indices = held_indices[near_local]
        near_labels = _labels(categories, near_indices)
        fold_rows.append(
            {
                "held_group": held_group,
                "train_candidates": int(len(train_selected)),
                "calibration_candidates": int(len(calibration_indices)),
                "test_tp_bgfar_candidates": int(len(pair_indices)),
                "selected_epoch": selected_epoch,
                "beta": calibration["beta"],
                "raw_auc_tp_bgfar": raw_auc,
                "prototype_auc_tp_bgfar": proto_auc,
                "joint_auc_tp_bgfar": joint_auc,
                "joint_gain_tp_bgfar": joint_auc - raw_auc,
                "raw_auc_tp_near_miss": _binary_roc_auc(
                    near_labels, raw_margin[near_indices]
                ),
                "prototype_auc_tp_near_miss": _binary_roc_auc(
                    near_labels, held_proto[near_local]
                ),
                "joint_auc_tp_near_miss": _binary_roc_auc(
                    near_labels, held_joint[near_local]
                ),
                "conditional_prototype_auc": _conditional_auc(
                    labels,
                    raw_margin[pair_indices],
                    held_proto[pair_local],
                    groups[pair_indices],
                    args.conditional_score_bins,
                ),
            }
        )
        print(
            f"[Candidate-conditioned-LOSO] held={held_group}, "
            f"raw/proto/joint={raw_auc:.6f}/{proto_auc:.6f}/{joint_auc:.6f}, "
            f"beta={calibration['beta']:.2f}",
            flush=True,
        )

    raw_macro = float(np.mean([row["raw_auc_tp_bgfar"] for row in fold_rows]))
    proto_values = [row["prototype_auc_tp_bgfar"] for row in fold_rows]
    proto_macro = float(np.mean(proto_values))
    proto_min = float(np.min(proto_values))
    joint_macro = float(np.mean([row["joint_auc_tp_bgfar"] for row in fold_rows]))
    conditional_macro = float(
        np.nanmean([row["conditional_prototype_auc"] for row in fold_rows])
    )
    raw_near_macro = float(
        np.nanmean([row["raw_auc_tp_near_miss"] for row in fold_rows])
    )
    joint_near_macro = float(
        np.nanmean([row["joint_auc_tp_near_miss"] for row in fold_rows])
    )
    gate = candidate_conditioned_gate(
        prototype_macro_auc=proto_macro,
        prototype_min_slide_auc=proto_min,
        joint_macro_auc=joint_macro,
        raw_macro_auc=raw_macro,
        conditional_macro_auc=conditional_macro,
        joint_near_miss_macro_auc=joint_near_macro,
        raw_near_miss_macro_auc=raw_near_macro,
    )
    audit = {
        "version": AUDIT_VERSION,
        "gate": gate,
        "checkpoint": manifest.get("checkpoint", ""),
        "checkpoint_sha256": manifest.get("checkpoint_sha256", ""),
        "archive": os.path.abspath(args.archive),
        "manifest": os.path.abspath(args.manifest),
        "feature_mode": args.feature_mode,
        "feature_source": feature_source,
        "training_categories": ["tp", "background_far_fp"],
        "training_weighting": "inverse_selected_candidates_per_slide",
        "excluded_from_training": [
            "near_miss_fp",
            "duplicate_fp",
            "low_score_fn",
            "empty_image_fp",
        ],
        "configuration": vars(args),
        "selected_epochs": selected_epochs,
    }
    _write_csv(os.path.join(args.output_dir, "loso_fold_metrics.csv"), fold_rows)
    _write_csv(
        os.path.join(args.output_dir, "loso_candidate_scores.csv"), candidate_rows
    )
    _write_csv(
        os.path.join(args.output_dir, "score_matching_audit.csv"), score_match_audit
    )
    _write_json(
        os.path.join(args.output_dir, "candidate_conditioned_proto_audit.json"), audit
    )

    if gate["gate_pass"]:
        final_indices, final_sampling = score_matched_group_sample(
            eligible,
            categories,
            groups,
            raw_margin,
            score_bins=args.score_bins,
            max_per_class_per_bin=args.max_per_class_per_bin,
            seed=args.seed + 900001,
        )
        final_epochs = max(1, int(round(float(np.median(selected_epochs)))))
        final_model, final_history = _fit_model(
            features,
            categories,
            groups,
            final_indices,
            np.zeros((0,), dtype=np.int64),
            args,
            device,
            fixed_epochs=final_epochs,
        )
        final_proto = _score_model(
            final_model, features[eligible], args.batch_size
        )
        final_calibration = _fit_joint_calibration(
            raw_margin[eligible],
            final_proto,
            _labels(categories, eligible),
            groups[eligible],
            beta_grid,
        )
        metadata = {
            "gate_pass": True,
            "feature_mode": args.feature_mode,
            "feature_source": feature_source,
            "training_categories": ["tp", "background_far_fp"],
            "training_weighting": "inverse_selected_candidates_per_slide",
            "checkpoint": manifest.get("checkpoint", ""),
            "checkpoint_sha256": manifest.get("checkpoint_sha256", ""),
            "source_archive": os.path.abspath(args.archive),
            "source_manifest": os.path.abspath(args.manifest),
            "selected_epochs": final_epochs,
            "training_candidates": int(len(final_indices)),
            "gate": gate,
            "source_train_calibration": {
                **final_calibration,
                "selection_labels": ["tp", "background_far_fp"],
                "selection_scope": "source_train_only",
            },
            "sampling": {
                "strategy": "slide_and_raw_score_bin_balanced",
                "score_bins": args.score_bins,
                "max_per_class_per_bin": args.max_per_class_per_bin,
            },
        }
        payload = final_model.export_payload(metadata)
        bank_path = os.path.join(
            args.output_dir, "candidate_conditioned_proto_bank.pth"
        )
        temporary = bank_path + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, bank_path)
        _write_csv(
            os.path.join(args.output_dir, "final_training_history.csv"),
            final_history,
        )
        _write_csv(
            os.path.join(args.output_dir, "final_score_matching_audit.csv"),
            final_sampling,
        )
        print(f"[Candidate-conditioned-LOSO] bank={bank_path}", flush=True)
    else:
        print(
            "[Candidate-conditioned-LOSO] gate failed; no training bank exported: "
            + ",".join(gate["failures"]),
            flush=True,
        )
        if args.require_gate:
            raise RuntimeError("candidate-conditioned prototype gate failed")
    print(f"[Candidate-conditioned-LOSO] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
