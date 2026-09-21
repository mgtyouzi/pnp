import hashlib
import math
import os
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SSCP_SELECTIVE_RESCUE_VERSION = "sscp_precision_budget_rescue_v2_20260903"
MINIMUM_MEANINGFUL_F1_GAIN = 0.002


def calculate_promotion_budget(
    raw_positive_count: int,
    ratio: float = 0.02,
    minimum: int = 1,
    maximum: int = 8,
) -> int:
    if raw_positive_count < 0 or not 0 < ratio <= 1:
        raise ValueError("invalid promotion budget inputs")
    if not 0 <= minimum <= maximum:
        raise ValueError("invalid promotion budget bounds")
    scaled = int(math.ceil(int(raw_positive_count) * float(ratio)))
    return max(int(minimum), min(int(maximum), scaled))


def prototype_bank_state_sha256(bank_payload: Dict[str, object]) -> str:
    version = str(bank_payload.get("implementation_version", ""))
    if not version:
        raise RuntimeError("prototype bank implementation version is missing")
    digest = hashlib.sha256()
    digest.update(version.encode("utf-8"))
    for key in ("foreground_prototypes", "background_prototypes"):
        value = bank_payload.get(key)
        if value is None:
            raise RuntimeError(f"prototype bank is missing {key}")
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().float()
            tensor = tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-6)
            array = tensor.numpy()
        else:
            array = np.asarray(value, dtype=np.float32)
            norm = np.linalg.norm(array, axis=1, keepdims=True)
            array = array / np.maximum(norm, 1e-6)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise RuntimeError(f"prototype bank contains invalid {key}")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_experiment_contract(
    training_args: Dict[str, object],
    initialization: Dict[str, object],
    bank_payload: Dict[str, object],
) -> Dict[str, object]:
    bank_id = str(bank_payload.get("bank_id", ""))
    initialized_bank_id = str(initialization.get("prototype_bank_id", ""))
    if not bank_id or initialized_bank_id != bank_id:
        raise RuntimeError(
            "prototype bank_id mismatch between checkpoint initialization and bank"
        )
    expected_state_hash = str(
        initialization.get("prototype_fixed_state_sha256", "")
    )
    actual_state_hash = prototype_bank_state_sha256(bank_payload)
    if not expected_state_hash or expected_state_hash != actual_state_hash:
        raise RuntimeError(
            "prototype bank state hash mismatch between checkpoint initialization "
            "and supplied bank"
        )
    expected_mode = "source_supervised_candidate_proto"
    if not bool(initialization.get("prototype_enabled", False)):
        raise RuntimeError("checkpoint initialization did not enable prototypes")
    for source, value in (
        ("checkpoint args", training_args.get("proto_mode")),
        ("checkpoint initialization", initialization.get("prototype_mode")),
    ):
        if str(value) != expected_mode:
            raise RuntimeError(f"{source} prototype mode mismatch: {value}")

    foreground_shape = tuple(bank_payload["foreground_prototypes"].shape)
    background_shape = tuple(bank_payload["background_prototypes"].shape)
    expected_fg = int(training_args.get("proto_num_fg", -1))
    expected_bg = int(training_args.get("proto_num_bg", -1))
    if foreground_shape[0] != expected_fg or background_shape[0] != expected_bg:
        raise RuntimeError("checkpoint prototype counts do not match the bank")
    if foreground_shape[1:] != background_shape[1:]:
        raise RuntimeError("foreground/background prototype dimensions differ")

    temperature = float(training_args.get("proto_temperature", 0.0))
    foreground_queue_size = int(training_args.get("proto_fg_queue_size", 0))
    background_queue_size = int(training_args.get("proto_bg_queue_size", 0))
    if temperature <= 0:
        raise RuntimeError("checkpoint prototype temperature must be positive")
    if foreground_queue_size < expected_fg or background_queue_size < expected_bg:
        raise RuntimeError("checkpoint prototype queue sizes are invalid")
    return {
        "prototype_mode": expected_mode,
        "prototype_bank_id": bank_id,
        "prototype_bank_state_sha256": actual_state_hash,
        "prototype_bank_path_at_training": str(
            initialization.get("prototype_bank_path", "")
        ),
        "prototype_temperature": temperature,
        "num_foreground_prototypes": expected_fg,
        "num_background_prototypes": expected_bg,
        "foreground_queue_size": foreground_queue_size,
        "background_queue_size": background_queue_size,
    }


def audit_dataset_split(
    train_files: Sequence[str],
    test_files: Sequence[str],
    group_fn: Callable[[str], str],
) -> Dict[str, object]:
    train_names = {os.path.basename(str(value)) for value in train_files}
    test_names = {os.path.basename(str(value)) for value in test_files}
    exact_overlap = sorted(train_names & test_names)
    if exact_overlap:
        raise RuntimeError(
            f"exact file overlap between train and test: {exact_overlap[:5]}"
        )
    train_groups = {str(group_fn(value)) for value in train_files}
    test_groups = {str(group_fn(value)) for value in test_files}
    group_overlap = sorted(train_groups & test_groups)
    return {
        "exact_file_overlap_count": 0,
        "slide_group_overlap_count": len(group_overlap),
        "overlapping_slide_groups": group_overlap,
        "source_test_group_disjoint": not group_overlap,
    }


def round_robin_group_indices(
    groups: Sequence[str], maximum: int = 0
) -> List[int]:
    buckets: Dict[str, List[int]] = {}
    for index, group in enumerate(groups):
        buckets.setdefault(str(group), []).append(index)
    limit = len(groups) if int(maximum) <= 0 else min(int(maximum), len(groups))
    offsets = {name: 0 for name in buckets}
    selected: List[int] = []
    while len(selected) < limit:
        added = False
        for name in sorted(buckets):
            offset = offsets[name]
            if offset >= len(buckets[name]):
                continue
            selected.append(buckets[name][offset])
            offsets[name] += 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _balanced_accuracy_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if not len(scores) or labels.all() or (~labels).all():
        raise ValueError("margin calibration needs positive and negative candidates")
    best = (-1.0, float("inf"))
    for threshold in np.unique(scores):
        prediction = scores >= threshold
        sensitivity = float(prediction[labels].mean())
        specificity = float((~prediction[~labels]).mean())
        candidate = (0.5 * (sensitivity + specificity), -float(threshold))
        if candidate > best:
            best = candidate
    return -best[1], best[0]


def fit_margin_calibration(
    labels: np.ndarray, prototype_margins: np.ndarray
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    margins = np.asarray(prototype_margins, dtype=np.float64).reshape(-1)
    if len(labels) != len(margins):
        raise ValueError("labels and prototype margins are misaligned")
    finite = np.isfinite(margins)
    labels = labels[finite]
    margins = margins[finite]
    if not len(margins) or not np.isin(labels, (0, 1)).all():
        raise ValueError("calibration values must be finite binary observations")
    threshold, balanced_accuracy = _balanced_accuracy_threshold(labels, margins)
    median = float(np.median(margins))
    mad = float(np.median(np.abs(margins - median)))
    robust_scale = 1.4826 * mad
    standard_scale = float(margins.std())
    scale = max(robust_scale, standard_scale * 0.25, 1e-6)
    return {
        "threshold": float(threshold),
        "scale": float(scale),
        "auc": _binary_auc(labels, margins),
        "balanced_accuracy": float(balanced_accuracy),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }


def fit_precision_constrained_calibration(
    labels: np.ndarray,
    prototype_margins: np.ndarray,
    groups: Sequence[str],
    minimum_precision: float = 0.60,
    minimum_group_precision: float = 0.50,
    minimum_predictions_per_group: int = 10,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    margins = np.asarray(prototype_margins, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).astype(str).reshape(-1)
    if not (len(labels) == len(margins) == len(groups)):
        raise ValueError("labels, prototype margins, and groups are misaligned")
    finite = np.isfinite(margins)
    labels, margins, groups = labels[finite], margins[finite], groups[finite]
    if (
        not len(labels)
        or not np.isin(labels, (0, 1)).all()
        or labels.min() == labels.max()
    ):
        raise ValueError("precision calibration needs finite positive and negative data")
    if not 0 < minimum_precision <= 1 or not 0 < minimum_group_precision <= 1:
        raise ValueError("precision constraints must be in (0, 1]")
    if int(minimum_predictions_per_group) < 1:
        raise ValueError("minimum_predictions_per_group must be positive")

    order = np.argsort(-margins, kind="mergesort")
    sorted_margins = margins[order]
    sorted_labels = labels[order]
    sorted_groups = groups[order]
    threshold_ends = np.flatnonzero(np.r_[
        sorted_margins[:-1] != sorted_margins[1:], True
    ])
    predicted_counts = threshold_ends + 1
    cumulative_true = np.cumsum(sorted_labels == 1)[threshold_ends]
    precisions = cumulative_true / predicted_counts
    valid = precisions >= float(minimum_precision)
    minimum_observed = np.ones(len(threshold_ends), dtype=np.float64)
    unique_groups = sorted(set(groups.tolist()))
    for group in unique_groups:
        group_mask = sorted_groups == group
        group_counts = np.cumsum(group_mask)[threshold_ends]
        group_true = np.cumsum(group_mask & (sorted_labels == 1))[threshold_ends]
        group_precision = np.divide(
            group_true,
            group_counts,
            out=np.zeros_like(group_true, dtype=np.float64),
            where=group_counts > 0,
        )
        valid &= group_counts >= int(minimum_predictions_per_group)
        valid &= group_precision >= float(minimum_group_precision)
        minimum_observed = np.minimum(minimum_observed, group_precision)

    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        raise ValueError("no prototype threshold satisfies precision constraints")
    selected_index = max(
        valid_indices.tolist(),
        key=lambda index: (
            int(cumulative_true[index]),
            float(precisions[index]),
            float(minimum_observed[index]),
            float(sorted_margins[threshold_ends[index]]),
        ),
    )
    true_positive_count = int(cumulative_true[selected_index])
    precision = float(precisions[selected_index])
    minimum_precision_observed = float(minimum_observed[selected_index])
    threshold = float(sorted_margins[threshold_ends[selected_index]])
    median = float(np.median(margins))
    mad = float(np.median(np.abs(margins - median)))
    scale = max(1.4826 * mad, float(margins.std()) * 0.25, 1e-6)
    predicted_count = int(predicted_counts[selected_index])
    return {
        "threshold": float(threshold),
        "scale": float(scale),
        "auc": _binary_auc(labels, margins),
        "precision": float(precision),
        "minimum_group_precision": minimum_precision_observed,
        "prediction_count": predicted_count,
        "true_positive_count": int(true_positive_count),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }


def selective_positive_rescue(
    raw_logits: np.ndarray,
    prototype_logits: np.ndarray,
    threshold: float,
    scale: float,
    uncertainty: float,
    strength: float,
    evidence_clip: float = 3.0,
    max_promotions: int = -1,
) -> Tuple[np.ndarray, Dict[str, int]]:
    raw = np.asarray(raw_logits, dtype=np.float64)
    prototype = np.asarray(prototype_logits, dtype=np.float64)
    if raw.shape != prototype.shape or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("raw and prototype logits must both have shape [N, 2]")
    if (
        not np.isfinite(threshold)
        or scale <= 0
        or uncertainty < 0
        or strength < 0
        or evidence_clip <= 0
    ):
        raise ValueError("invalid selective rescue calibration")

    raw_margin = raw[:, 0] - raw[:, 1]
    prototype_margin = prototype[:, 0] - prototype[:, 1]
    raw_background = raw_margin < 0.0
    uncertain = raw_margin >= -float(uncertainty)
    evidence = np.clip(
        (prototype_margin - float(threshold)) / float(scale),
        0.0,
        float(evidence_clip),
    )
    eligible_before_budget = raw_background & uncertain & (evidence > 0.0)
    eligible = eligible_before_budget.copy()
    if int(max_promotions) >= 0 and int(eligible.sum()) > int(max_promotions):
        candidate_indices = np.flatnonzero(eligible)
        order = np.argsort(-evidence[candidate_indices], kind="mergesort")
        keep = candidate_indices[order[:int(max_promotions)]]
        eligible[:] = False
        eligible[keep] = True
    correction = np.zeros(len(raw), dtype=np.float64)
    correction[eligible] = float(strength) * evidence[eligible]

    fused = raw.copy()
    fused[:, 0] += 0.5 * correction
    fused[:, 1] -= 0.5 * correction
    raw_classes = np.argmax(raw, axis=-1)
    fused_classes = np.argmax(fused, axis=-1)
    return fused, {
        "eligible_before_budget": int(eligible_before_budget.sum()),
        "eligible": int(eligible.sum()),
        "budget": int(max_promotions),
        "promoted": int(((raw_classes == 1) & (fused_classes == 0)).sum()),
        "background_to_cell": int(((raw_classes == 1) & (fused_classes == 0)).sum()),
        "cell_to_background": int(((raw_classes == 0) & (fused_classes == 1)).sum()),
    }


def select_rescue_configuration(
    rows: Iterable[Dict[str, float]],
) -> Dict[str, object]:
    rows: List[Dict[str, float]] = [dict(row) for row in rows]
    controls = [
        row for row in rows
        if float(row["uncertainty"]) == 0.0 and float(row["strength"]) == 0.0
    ]
    candidates = [
        row for row in rows
        if float(row["uncertainty"]) > 0.0 and float(row["strength"]) > 0.0
    ]
    if len(controls) != 1 or not candidates:
        raise ValueError("configuration selection needs identity and rescue rows")
    baseline = controls[0]
    best = max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row.get("background_to_cell", 0.0)),
            -float(row["uncertainty"]),
            -float(row["strength"]),
        ),
    )
    gain = float(best["f1"]) - float(baseline["f1"])
    selected = best if gain > 0 else baseline
    return {
        **selected,
        "status": "selected" if gain > 0 else "no_calibration_gain",
        "calibration_baseline_f1": float(baseline["f1"]),
        "calibration_f1_gain": max(gain, 0.0),
    }


def select_precision_constrained_configuration(
    rows: Iterable[Dict[str, float]],
    minimum_promotion_precision: float = 0.60,
    minimum_slides_improved_fraction: float = 0.60,
) -> Dict[str, object]:
    rows = [dict(row) for row in rows]
    controls = [
        row for row in rows
        if float(row["uncertainty"]) == 0.0 and float(row["strength"]) == 0.0
    ]
    if len(controls) != 1:
        raise ValueError("configuration selection needs one identity row")
    baseline = controls[0]
    candidates = [
        row for row in rows
        if float(row["uncertainty"]) > 0.0
        and float(row["strength"]) > 0.0
        and float(row.get("promotion_precision", 0.0))
        >= float(minimum_promotion_precision)
        and float(row.get("promoted_far_background", float("inf")))
        <= float(row.get("promoted_positive_candidate", 0.0))
        and float(row.get("slides_improved_fraction", 0.0))
        >= float(minimum_slides_improved_fraction)
    ]
    if not candidates:
        return {
            **baseline,
            "status": "no_precision_constrained_gain",
            "calibration_baseline_f1": float(baseline["f1"]),
            "calibration_f1_gain": 0.0,
        }
    best = max(
        candidates,
        key=lambda row: (
            float(row.get("macro_f1", row["f1"])),
            float(row["f1"]),
            float(row["precision"]),
            -float(row.get("background_to_cell", 0.0)),
        ),
    )
    f1_gain = float(best["f1"]) - float(baseline["f1"])
    macro_gain = float(best.get("macro_f1", best["f1"])) - float(
        baseline.get("macro_f1", baseline["f1"])
    )
    if f1_gain <= 0 or macro_gain <= 0:
        return {
            **baseline,
            "status": "no_precision_constrained_gain",
            "calibration_baseline_f1": float(baseline["f1"]),
            "calibration_f1_gain": 0.0,
        }
    return {
        **best,
        "status": "selected",
        "calibration_baseline_f1": float(baseline["f1"]),
        "calibration_f1_gain": float(f1_gain),
        "calibration_macro_f1_gain": float(macro_gain),
    }


def decide_fixed_rescue(
    baseline: Dict[str, float],
    fixed: Dict[str, float],
    minimum_f1_gain: float = 0.002,
    minimum_promotion_precision: float = 0.60,
) -> Dict[str, object]:
    if float(minimum_f1_gain) < MINIMUM_MEANINGFUL_F1_GAIN:
        raise ValueError(
            "minimum_f1_gain cannot be lower than "
            f"{MINIMUM_MEANINGFUL_F1_GAIN}"
        )
    gain = float(fixed["f1"]) - float(baseline["f1"])
    stable_count = 0.7 <= float(fixed["pred_gt_ratio"]) <= 1.5
    has_promotion_metrics = "promotion_precision" in fixed
    promotion_precision = float(fixed.get("promotion_precision", 1.0))
    positive_promotions = float(fixed.get("promoted_positive_candidate", 0.0))
    far_promotions = float(fixed.get("promoted_far_background", 0.0))
    promotion_gate_pass = bool(
        not has_promotion_metrics
        or (
            promotion_precision >= float(minimum_promotion_precision)
            and far_promotions <= positive_promotions
        )
    )
    gate_pass = bool(
        gain >= minimum_f1_gain and stable_count and promotion_gate_pass
    )
    return {
        "gate_pass": gate_pass,
        "minimum_f1_gain": float(minimum_f1_gain),
        "minimum_promotion_precision": float(minimum_promotion_precision),
        "promotion_precision": promotion_precision,
        "promotion_gate_pass": promotion_gate_pass,
        "f1_gain": gain,
        "precision_delta": float(fixed["precision"]) - float(baseline["precision"]),
        "recall_delta": float(fixed["recall"]) - float(baseline["recall"]),
        "pred_gt_ratio": float(fixed["pred_gt_ratio"]),
        "fp_background_far_delta": float(fixed["fp_background_far"])
        - float(baseline["fp_background_far"]),
        "fn_low_score_or_background_delta": float(
            fixed["fn_low_score_or_background"]
        ) - float(baseline["fn_low_score_or_background"]),
        "next_action": (
            "integrate_fixed_selective_rescue"
            if gate_pass
            else "stop_current_sscp_prototype"
        ),
    }
