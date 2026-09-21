from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


CANDIDATE_CONDITIONED_DATA_VERSION = (
    "candidate_conditioned_proto_data_v2_20260829"
)

FEATURE_MODE_TO_SOURCE = {
    "cls_only": "p2p_cls_features_before_final_classifier",
    "cls_reg_context": "p2p_cls_reg_attention_offset_context",
    "spatial_morphology": (
        "p2p_fpn_p2_p3_local_spatial_center_ring_contrast"
    ),
}

CATEGORY_TP = 0
CATEGORY_BACKGROUND_FAR_FP = 1
CATEGORY_NEAR_MISS_FP = 2
CATEGORY_DUPLICATE_FP = 3
CATEGORY_LOW_SCORE_FN = 4
CATEGORY_EMPTY_IMAGE_FP = 5

CATEGORY_NAME_TO_ID = {
    "tp": CATEGORY_TP,
    "background_far_fp": CATEGORY_BACKGROUND_FAR_FP,
    "near_miss_fp": CATEGORY_NEAR_MISS_FP,
    "duplicate_fp": CATEGORY_DUPLICATE_FP,
    "low_score_fn": CATEGORY_LOW_SCORE_FN,
    "empty_image_fp": CATEGORY_EMPTY_IMAGE_FP,
}
CATEGORY_ID_TO_NAME = {value: key for key, value in CATEGORY_NAME_TO_ID.items()}


@dataclass(frozen=True)
class LosoImageSplit:
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denominator, 1e-6)


def build_candidate_feature_matrix(
    data: Dict[str, np.ndarray], feature_mode: str
) -> Tuple[np.ndarray, str]:
    if feature_mode not in FEATURE_MODE_TO_SOURCE:
        raise ValueError(f"unsupported candidate feature mode: {feature_mode}")
    if feature_mode == "spatial_morphology":
        if "morphology_features" not in data:
            raise RuntimeError("candidate archive lacks morphology features")
        morphology = _row_l2_normalize(data["morphology_features"])
        if not np.isfinite(morphology).all():
            raise RuntimeError("morphology features contain non-finite values")
        return morphology, FEATURE_MODE_TO_SOURCE[feature_mode]
    if "features" not in data:
        raise RuntimeError("candidate archive lacks classification features")
    classification = _row_l2_normalize(data["features"])
    if feature_mode == "cls_only":
        return classification, FEATURE_MODE_TO_SOURCE[feature_mode]

    required = {"reg_features", "cls_attn", "reg_attn", "reg_offset"}
    missing = sorted(required.difference(data))
    if missing:
        raise RuntimeError(f"candidate archive lacks context arrays: {missing}")
    components = [
        classification,
        _row_l2_normalize(data["reg_features"]),
        np.asarray(data["cls_attn"], dtype=np.float32),
        np.asarray(data["reg_attn"], dtype=np.float32),
        np.clip(np.asarray(data["reg_offset"], dtype=np.float32) / 32.0, -2.0, 2.0),
    ]
    lengths = {len(component) for component in components}
    if len(lengths) != 1:
        raise RuntimeError("candidate context arrays are misaligned")
    if not all(component.ndim == 2 for component in components):
        raise RuntimeError("candidate context arrays must be matrices")
    matrix = np.concatenate(components, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("candidate context features contain non-finite values")
    return matrix, FEATURE_MODE_TO_SOURCE[feature_mode]


def build_loso_image_split(
    group_index: np.ndarray,
    image_index: np.ndarray,
    held_group: int,
    calibration_fraction: float,
    seed: int,
) -> LosoImageSplit:
    groups = np.asarray(group_index, dtype=np.int64).reshape(-1)
    images = np.asarray(image_index, dtype=np.int64).reshape(-1)
    if groups.shape != images.shape:
        raise ValueError("group_index and image_index must be aligned")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if held_group not in set(groups.tolist()):
        raise ValueError(f"held_group={held_group} is absent")

    train_mask = np.zeros(len(groups), dtype=bool)
    calibration_mask = np.zeros(len(groups), dtype=bool)
    test_mask = groups == int(held_group)
    for group in sorted(int(value) for value in np.unique(groups)):
        if group == held_group:
            continue
        local_mask = groups == group
        local_images = np.unique(images[local_mask])
        if len(local_images) < 2:
            raise RuntimeError(
                f"group={group} needs at least two images for calibration"
            )
        count = max(1, int(round(len(local_images) * calibration_fraction)))
        count = min(count, len(local_images) - 1)
        random = np.random.RandomState(int(seed) + 1009 * group)
        calibration_images = random.choice(local_images, size=count, replace=False)
        selected_calibration = local_mask & np.isin(images, calibration_images)
        calibration_mask |= selected_calibration
        train_mask |= local_mask & ~selected_calibration

    split = LosoImageSplit(
        train_indices=np.flatnonzero(train_mask),
        calibration_indices=np.flatnonzero(calibration_mask),
        test_indices=np.flatnonzero(test_mask),
    )
    train_images = set(images[split.train_indices].tolist())
    calibration_images = set(images[split.calibration_indices].tolist())
    test_images = set(images[split.test_indices].tolist())
    if train_images.intersection(calibration_images):
        raise RuntimeError("train/calibration image leakage")
    if test_images.intersection(train_images | calibration_images):
        raise RuntimeError("held-out image leakage")
    if int(held_group) in set(groups[split.train_indices].tolist()):
        raise RuntimeError("held-out group leaked into training")
    if int(held_group) in set(groups[split.calibration_indices].tolist()):
        raise RuntimeError("held-out group leaked into calibration")
    return split


def _deterministic_take(
    indices: np.ndarray, count: int, seed: int
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if count >= len(indices):
        return np.sort(indices)
    random = np.random.RandomState(int(seed))
    return np.sort(random.choice(indices, size=int(count), replace=False))


def group_balanced_sample_weights(
    group_index: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    groups = np.asarray(group_index, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(indices) and (indices.min() < 0 or indices.max() >= len(groups)):
        raise IndexError("candidate index is out of range")
    selected_groups = groups[indices]
    if not len(selected_groups):
        return np.zeros((0,), dtype=np.float32)
    unique, counts = np.unique(selected_groups, return_counts=True)
    count_by_group = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray(
        [1.0 / count_by_group[int(group)] for group in selected_groups],
        dtype=np.float64,
    )
    weights /= weights.mean()
    return weights.astype(np.float32)


def score_matched_group_sample(
    indices: np.ndarray,
    category: np.ndarray,
    group_index: np.ndarray,
    raw_margin: np.ndarray,
    score_bins: int,
    max_per_class_per_bin: int,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    categories = np.asarray(category, dtype=np.int64).reshape(-1)
    groups = np.asarray(group_index, dtype=np.int64).reshape(-1)
    margins = np.asarray(raw_margin, dtype=np.float64).reshape(-1)
    if not (len(categories) == len(groups) == len(margins)):
        raise ValueError("candidate arrays must be aligned")
    if score_bins < 1 or max_per_class_per_bin < 1:
        raise ValueError("score_bins and per-bin cap must be positive")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(categories)):
        raise IndexError("candidate index is out of range")

    eligible = indices[
        np.isin(
            categories[indices],
            [CATEGORY_TP, CATEGORY_BACKGROUND_FAR_FP],
        )
        & np.isfinite(margins[indices])
    ]
    selected_parts: List[np.ndarray] = []
    audit: List[Dict[str, int]] = []
    for group in sorted(int(value) for value in np.unique(groups[eligible])):
        local = eligible[groups[eligible] == group]
        if not len(local):
            continue
        quantiles = np.linspace(0.0, 1.0, int(score_bins) + 1)
        edges = np.unique(np.quantile(margins[local], quantiles))
        interior = edges[1:-1] if len(edges) > 2 else np.asarray([])
        local_bins = np.digitize(margins[local], interior, right=False)
        for bin_index in sorted(int(value) for value in np.unique(local_bins)):
            in_bin = local[local_bins == bin_index]
            positive = in_bin[categories[in_bin] == CATEGORY_TP]
            negative = in_bin[
                categories[in_bin] == CATEGORY_BACKGROUND_FAR_FP
            ]
            count = min(
                len(positive),
                len(negative),
                int(max_per_class_per_bin),
            )
            row = {
                "group": group,
                "score_bin": bin_index,
                "available_tp": int(len(positive)),
                "available_background_far_fp": int(len(negative)),
                "selected_tp": int(count),
                "selected_background_far_fp": int(count),
            }
            audit.append(row)
            if count == 0:
                continue
            selected_parts.append(
                _deterministic_take(
                    positive,
                    count,
                    seed + 100003 * group + 1009 * bin_index,
                )
            )
            selected_parts.append(
                _deterministic_take(
                    negative,
                    count,
                    seed + 200003 * group + 2017 * bin_index,
                )
            )
    selected = (
        np.sort(np.concatenate(selected_parts))
        if selected_parts
        else np.zeros((0,), dtype=np.int64)
    )
    if len(selected):
        selected_categories = set(categories[selected].tolist())
        if not selected_categories.issubset(
            {CATEGORY_TP, CATEGORY_BACKGROUND_FAR_FP}
        ):
            raise RuntimeError("non-training candidate leaked into prototype fitting")
    return selected, audit


def candidate_conditioned_gate(
    *,
    prototype_macro_auc: float,
    prototype_min_slide_auc: float,
    joint_macro_auc: float,
    raw_macro_auc: float,
    conditional_macro_auc: float,
    joint_near_miss_macro_auc: float,
    raw_near_miss_macro_auc: float,
    minimum_prototype_macro_auc: float = 0.65,
    minimum_prototype_min_slide_auc: float = 0.55,
    minimum_joint_gain: float = 0.02,
    minimum_conditional_macro_auc: float = 0.60,
    maximum_near_miss_drop: float = 0.01,
) -> Dict[str, object]:
    metrics = {
        "prototype_macro_auc": float(prototype_macro_auc),
        "prototype_min_slide_auc": float(prototype_min_slide_auc),
        "joint_macro_auc": float(joint_macro_auc),
        "raw_macro_auc": float(raw_macro_auc),
        "conditional_macro_auc": float(conditional_macro_auc),
        "joint_near_miss_macro_auc": float(joint_near_miss_macro_auc),
        "raw_near_miss_macro_auc": float(raw_near_miss_macro_auc),
    }
    failures = []
    if not np.isfinite(metrics["prototype_macro_auc"]) or (
        metrics["prototype_macro_auc"] < minimum_prototype_macro_auc
    ):
        failures.append("prototype_macro_auc")
    if not np.isfinite(metrics["prototype_min_slide_auc"]) or (
        metrics["prototype_min_slide_auc"] < minimum_prototype_min_slide_auc
    ):
        failures.append("prototype_min_slide_auc")
    joint_gain = metrics["joint_macro_auc"] - metrics["raw_macro_auc"]
    if not np.isfinite(joint_gain) or joint_gain < minimum_joint_gain:
        failures.append("joint_gain")
    if not np.isfinite(metrics["conditional_macro_auc"]) or (
        metrics["conditional_macro_auc"] < minimum_conditional_macro_auc
    ):
        failures.append("conditional_macro_auc")
    near_miss_drop = (
        metrics["raw_near_miss_macro_auc"]
        - metrics["joint_near_miss_macro_auc"]
    )
    if not np.isfinite(near_miss_drop) or near_miss_drop > maximum_near_miss_drop:
        failures.append("near_miss_drop")
    return {
        "gate_pass": not failures,
        "failures": failures,
        "metrics": {
            **metrics,
            "joint_gain": float(joint_gain),
            "near_miss_drop": float(near_miss_drop),
        },
        "thresholds": {
            "minimum_prototype_macro_auc": float(minimum_prototype_macro_auc),
            "minimum_prototype_min_slide_auc": float(
                minimum_prototype_min_slide_auc
            ),
            "minimum_joint_gain": float(minimum_joint_gain),
            "minimum_conditional_macro_auc": float(
                minimum_conditional_macro_auc
            ),
            "maximum_near_miss_drop": float(maximum_near_miss_drop),
        },
    }
