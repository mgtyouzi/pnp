from typing import Dict, Iterable, List, Tuple

import numpy as np


COUNT_KEYS = (
    "images",
    "gt",
    "pred",
    "tp",
    "fp",
    "fn",
    "fn_no_candidate",
    "fn_low_score_or_background",
    "fn_localization_shift",
    "fn_near_candidate_unmatched",
    "fp_empty_image",
    "fp_duplicate",
    "fp_near_miss",
    "fp_background_far",
)

SUPPRESSION_VERSION = "frozen_proto_one_way_suppression_v1_20260828"
CANDIDATE_AUDIT_VERSION = "frozen_proto_candidate_separation_v1_20260828"


def validate_candidate_audit_calibration(
    calibration: Dict[str, object],
    checkpoint_sha256: str,
    bank_id: str,
    bank_state_sha256: str,
) -> None:
    expected = {
        "version": CANDIDATE_AUDIT_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "bank_id": bank_id,
        "bank_state_sha256": bank_state_sha256,
    }
    mismatches = [
        f"{key}: expected={value}, got={calibration.get(key)}"
        for key, value in expected.items()
        if calibration.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "candidate calibration identity mismatch: " + "; ".join(mismatches)
        )


def load_mean_std(path: str) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.shape != (2, 3):
        raise ValueError(
            f"mean/std file must have shape [2, 3], got {values.shape}: {path}"
        )
    mean, std = values
    if not np.isfinite(values).all() or np.any(std <= 0):
        raise ValueError(f"mean/std file contains invalid values: {path}")
    return mean, std


def validate_checkpoint_load_report(report: Dict[str, object]) -> None:
    missing_detector_keys = [
        key
        for key in report.get("missing_model_keys", [])
        if not str(key).startswith("prototype_head.")
    ]
    failures = []
    if int(report.get("skipped_count", 0)):
        failures.append(f"skipped_keys={report.get('skipped_keys', [])}")
    if report.get("shape_mismatches"):
        failures.append(f"shape_mismatches={report['shape_mismatches']}")
    if report.get("unexpected_checkpoint_keys"):
        failures.append(
            f"unexpected_checkpoint_keys={report['unexpected_checkpoint_keys']}"
        )
    if missing_detector_keys:
        failures.append(f"missing_detector_keys={missing_detector_keys}")
    if report.get("unexpected_model_keys"):
        failures.append(f"unexpected_model_keys={report['unexpected_model_keys']}")
    if failures:
        raise RuntimeError(
            "detector checkpoint load is incomplete: " + "; ".join(failures)
        )


def resolve_csv_fieldnames(
    rows: List[Dict[str, object]], fieldnames: Iterable[str] = None
) -> List[str]:
    if rows:
        resolved = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    resolved.append(key)
        return resolved
    if fieldnames is None:
        raise ValueError("empty CSV requires explicit fieldnames")
    return list(fieldnames)


def parse_alphas(value: str) -> List[float]:
    alphas = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("fusion alphas must be non-negative")
    if 0.0 not in alphas:
        raise ValueError("fusion sweep must contain alpha=0 identity control")
    return alphas


def parse_quantiles(value: str) -> List[float]:
    quantiles = sorted(
        set(float(item.strip()) for item in value.split(",") if item.strip())
    )
    if not quantiles or quantiles[0] < 0 or quantiles[-1] > 1:
        raise ValueError("quantiles must be in [0, 1]")
    return quantiles


def fuse_binary_logits(
    raw_logits: np.ndarray,
    prototype_logits: np.ndarray,
    alpha: float,
    clip: float,
) -> np.ndarray:
    raw = np.asarray(raw_logits, dtype=np.float64)
    prototype = np.asarray(prototype_logits, dtype=np.float64)
    if raw.shape != prototype.shape or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("raw and prototype logits must both have shape [N, 2]")
    if alpha < 0 or clip <= 0:
        raise ValueError("alpha must be non-negative and clip must be positive")
    if alpha == 0:
        return raw.copy()
    margin = np.clip(prototype[:, 0] - prototype[:, 1], -clip, clip)
    correction = alpha * margin
    fused = raw.copy()
    fused[:, 0] += 0.5 * correction
    fused[:, 1] -= 0.5 * correction
    return fused


def suppress_raw_cell_logits(
    raw_logits: np.ndarray,
    prototype_logits: np.ndarray,
    alpha: float,
    threshold: float,
    clip: float,
) -> np.ndarray:
    raw = np.asarray(raw_logits, dtype=np.float64)
    prototype = np.asarray(prototype_logits, dtype=np.float64)
    if raw.shape != prototype.shape or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("raw and prototype logits must both have shape [N, 2]")
    if alpha < 0 or clip <= 0 or not np.isfinite(threshold):
        raise ValueError(
            "alpha must be non-negative; threshold finite; clip positive"
        )
    if alpha == 0:
        return raw.copy()

    raw_cell = np.argmax(raw, axis=-1) == 0
    prototype_margin = prototype[:, 0] - prototype[:, 1]
    penalty = alpha * np.clip(threshold - prototype_margin, 0.0, clip)
    fused = raw.copy()
    fused[raw_cell, 0] -= 0.5 * penalty[raw_cell]
    fused[raw_cell, 1] += 0.5 * penalty[raw_cell]
    return fused


def thresholds_from_raw_cell_quantiles(
    raw_logits: np.ndarray,
    prototype_logits: np.ndarray,
    quantiles: Iterable[float],
) -> List[float]:
    raw = np.asarray(raw_logits, dtype=np.float64)
    prototype = np.asarray(prototype_logits, dtype=np.float64)
    if raw.shape != prototype.shape or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("raw and prototype logits must both have shape [N, 2]")
    quantiles = sorted(set(float(value) for value in quantiles))
    if not quantiles or quantiles[0] < 0 or quantiles[-1] > 1:
        raise ValueError("threshold quantiles must be in [0, 1]")
    raw_cell = np.argmax(raw, axis=-1) == 0
    margins = prototype[raw_cell, 0] - prototype[raw_cell, 1]
    margins = margins[np.isfinite(margins)]
    if not len(margins):
        raise ValueError("cannot derive thresholds without raw-cell candidates")
    values = np.quantile(margins, quantiles)
    return sorted(set(float(value) for value in np.asarray(values).reshape(-1)))


def select_source_suppression(
    rows: Iterable[Dict[str, float]],
    max_f1_drop: float = 0.001,
    max_recall_drop: float = 0.005,
) -> Dict[str, object]:
    rows = [dict(row) for row in rows]
    controls = [row for row in rows if float(row["alpha"]) == 0.0]
    if len(controls) != 1:
        raise ValueError("source calibration needs exactly one alpha=0 control")
    if max_f1_drop < 0 or max_recall_drop < 0:
        raise ValueError("source safety tolerances must be non-negative")
    baseline = controls[0]
    minimum_f1 = float(baseline["f1"]) - max_f1_drop
    minimum_recall = float(baseline["recall"]) - max_recall_drop
    candidates = [
        row
        for row in rows
        if float(row["alpha"]) > 0
        and float(row["f1"]) >= minimum_f1
        and float(row["recall"]) >= minimum_recall
        and float(row["fp_background_far"])
        < float(baseline["fp_background_far"])
    ]
    if not candidates:
        return {
            "status": "no_safe_source_candidate",
            "alpha": 0.0,
            "threshold": None,
            "background_far_reduction": 0.0,
            "baseline_f1": float(baseline["f1"]),
            "baseline_recall": float(baseline["recall"]),
        }
    selected = min(
        candidates,
        key=lambda row: (
            float(row["fp_background_far"]),
            -float(row["f1"]),
            float(row["alpha"]),
            float(row["threshold"]),
        ),
    )
    return {
        "status": "selected",
        "alpha": float(selected["alpha"]),
        "threshold": float(selected["threshold"]),
        "background_far_reduction": float(baseline["fp_background_far"])
        - float(selected["fp_background_far"]),
        "baseline_f1": float(baseline["f1"]),
        "selected_f1": float(selected["f1"]),
        "baseline_recall": float(baseline["recall"]),
        "selected_recall": float(selected["recall"]),
    }


def decide_suppression_target(
    rows: Iterable[Dict[str, float]],
    minimum_f1_gain: float = 0.003,
    minimum_background_far_reduction_fraction: float = 0.05,
    maximum_recall_drop: float = 0.005,
    maximum_cls_fn_increase_fraction: float = 0.01,
) -> Dict[str, object]:
    rows = [dict(row) for row in rows]
    controls = [row for row in rows if row.get("label") == "identity"]
    fixed_rows = [row for row in rows if row.get("label") == "source_fixed"]
    if len(controls) != 1 or len(fixed_rows) != 1:
        raise ValueError("target decision needs identity and source_fixed rows")
    baseline, fixed = controls[0], fixed_rows[0]
    f1_gain = float(fixed["f1"]) - float(baseline["f1"])
    recall_drop = float(baseline["recall"]) - float(fixed["recall"])
    baseline_background_far = float(baseline["fp_background_far"])
    background_far_reduction = baseline_background_far - float(
        fixed["fp_background_far"]
    )
    background_far_fraction = (
        background_far_reduction / baseline_background_far
        if baseline_background_far
        else 0.0
    )
    cls_fn_delta = float(fixed["fn_low_score_or_background"]) - float(
        baseline["fn_low_score_or_background"]
    )
    cls_fn_limit = maximum_cls_fn_increase_fraction * float(baseline["gt"])
    no_background_promotion = float(fixed["background_to_cell"]) == 0.0
    promising = (
        no_background_promotion
        and f1_gain >= minimum_f1_gain
        and background_far_fraction >= minimum_background_far_reduction_fraction
        and recall_drop <= maximum_recall_drop
        and cls_fn_delta <= cls_fn_limit
    )
    return {
        "status": "promising" if promising else "no_detection_gain",
        "action": (
            "run_full_fixed_target_evaluation"
            if promising
            else "do_not_train_or_integrate_current_frozen_bank"
        ),
        "diagnostic_only": True,
        "f1_gain": f1_gain,
        "recall_drop": recall_drop,
        "background_far_reduction": background_far_reduction,
        "background_far_reduction_fraction": background_far_fraction,
        "cls_fn_delta": cls_fn_delta,
        "cls_fn_increase_limit": cls_fn_limit,
        "background_to_cell": float(fixed["background_to_cell"]),
    }


def validate_suppression_calibration(
    payload: Dict[str, object],
    checkpoint_sha256: str,
    bank_id: str,
    bank_state_sha256: str,
) -> None:
    failures = []
    if payload.get("version") != SUPPRESSION_VERSION:
        failures.append("version")
    selection = payload.get("selection", {})
    if not isinstance(selection, dict) or selection.get("status") != "selected":
        failures.append("selection")
    elif (
        float(selection.get("alpha", 0.0)) <= 0
        or selection.get("threshold") is None
    ):
        failures.append("selection_parameters")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        failures.append("checkpoint_sha256")
    if payload.get("bank_id") != bank_id:
        failures.append("bank_id")
    if payload.get("bank_state_sha256") != bank_state_sha256:
        failures.append("bank_state_sha256")
    if float(payload.get("fusion_clip", 0.0)) <= 0:
        failures.append("fusion_clip")
    if failures:
        raise RuntimeError(
            "suppression calibration identity mismatch: " + ", ".join(failures)
        )


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "mean": float(values.mean()),
    }


def distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.zeros((len(points_a), len(points_b)), dtype=np.float64)
    difference = points_a[:, None, :].astype(np.float64) - points_b[None, :, :].astype(np.float64)
    return np.sqrt(np.sum(difference * difference, axis=2))


def deduplicate_scores(
    points: np.ndarray, scores: np.ndarray, interval: float
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    fused = np.zeros(len(points), dtype=bool)
    kept_points = []
    kept_scores = []
    for index in range(len(points)):
        if fused[index]:
            continue
        local = np.where(
            np.linalg.norm(points[[index]] - points[index:], axis=1) < interval
        )[0] + index
        fused[local] = True
        local_scores = scores[local]
        row, _ = np.unravel_index(np.argmax(local_scores), local_scores.shape)
        kept_points.append(points[local[row]])
        kept_scores.append(scores[local[row]])
    if not kept_points:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, scores.shape[1]), dtype=np.float64)
    return np.stack(kept_points), np.stack(kept_scores)


def greedy_match(
    pred_points: np.ndarray,
    pred_scores: np.ndarray,
    gt_points: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int, float]], set, set]:
    if len(pred_points) == 0 or len(gt_points) == 0:
        return [], set(), set()
    order = np.argsort(-pred_scores)
    distances = distance_matrix(pred_points[order], gt_points)
    unmatched_gt = np.ones(len(gt_points), dtype=bool)
    matches = []
    for sorted_index, pred_index in enumerate(order):
        candidates = np.where(unmatched_gt)[0]
        if not len(candidates):
            break
        nearest = int(candidates[np.argmin(distances[sorted_index, candidates])])
        value = float(distances[sorted_index, nearest])
        if value <= threshold:
            matches.append((int(pred_index), nearest, value))
            unmatched_gt[nearest] = False
    return matches, {row[0] for row in matches}, {row[1] for row in matches}


def _deduplicated_candidate_indices(
    points: np.ndarray, scores: np.ndarray, interval: float
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    fused = np.zeros(len(points), dtype=bool)
    kept = []
    for index in range(len(points)):
        if fused[index]:
            continue
        local = np.where(
            np.linalg.norm(points[[index]] - points[index:], axis=1) < interval
        )[0] + index
        fused[local] = True
        row, _ = np.unravel_index(np.argmax(scores[local]), scores[local].shape)
        kept.append(int(local[row]))
    return np.asarray(kept, dtype=np.int64)


def label_candidate_diagnostics(
    points: np.ndarray,
    raw_logits: np.ndarray,
    prototype_logits: np.ndarray,
    gt_points: np.ndarray,
    dedup_interval: float,
    match_dis: float,
    near_radius: float,
) -> List[Dict[str, object]]:
    points = np.asarray(points, dtype=np.float64)
    raw_logits = np.asarray(raw_logits, dtype=np.float64)
    prototype_logits = np.asarray(prototype_logits, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    if (
        len(points) != len(raw_logits)
        or len(points) != len(prototype_logits)
        or raw_logits.shape != prototype_logits.shape
        or raw_logits.ndim != 2
        or raw_logits.shape[1] != 2
    ):
        raise ValueError("candidate points and binary logits are misaligned")

    probabilities = softmax(raw_logits)
    raw_classes = np.argmax(probabilities, axis=-1)
    reserved_indices = np.flatnonzero(raw_classes == 0)
    kept_local = _deduplicated_candidate_indices(
        points[reserved_indices], probabilities[reserved_indices], dedup_interval
    )
    kept_indices = reserved_indices[kept_local]
    pred_points = points[kept_indices]
    pred_cell_scores = probabilities[kept_indices, 0]
    matches, matched_pred, matched_gt = greedy_match(
        pred_points, pred_cell_scores, gt_points, match_dis
    )

    rows = []
    for pred_index, candidate_index in enumerate(kept_indices):
        if pred_index in matched_pred:
            category = "tp"
        elif not len(gt_points):
            category = "empty_image_fp"
        else:
            distances = distance_matrix(pred_points[[pred_index]], gt_points)[0]
            nearest_gt = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_gt])
            if nearest_distance <= match_dis and nearest_gt in matched_gt:
                category = "duplicate_fp"
            elif nearest_distance <= near_radius:
                category = "near_miss_fp"
            else:
                category = "background_far_fp"
        rows.append(
            {
                "category": category,
                "candidate_index": int(candidate_index),
                "raw_class": 0,
                "raw_cell_probability": float(probabilities[candidate_index, 0]),
                "raw_margin": float(
                    raw_logits[candidate_index, 0] - raw_logits[candidate_index, 1]
                ),
                "prototype_margin": float(
                    prototype_logits[candidate_index, 0]
                    - prototype_logits[candidate_index, 1]
                ),
            }
        )

    for gt_index, gt_point in enumerate(gt_points):
        if gt_index in matched_gt or not len(points):
            continue
        if len(pred_points):
            nearest_reserved = float(
                distance_matrix(pred_points, gt_point[None])[:, 0].min()
            )
            if match_dis < nearest_reserved <= near_radius:
                continue
        distances = distance_matrix(points, gt_point[None])[:, 0]
        candidate_index = int(np.argmin(distances))
        if distances[candidate_index] > near_radius or raw_classes[candidate_index] != 1:
            continue
        rows.append(
            {
                "category": "low_score_fn",
                "candidate_index": candidate_index,
                "gt_index": int(gt_index),
                "raw_class": 1,
                "raw_cell_probability": float(probabilities[candidate_index, 0]),
                "raw_margin": float(
                    raw_logits[candidate_index, 0] - raw_logits[candidate_index, 1]
                ),
                "prototype_margin": float(
                    prototype_logits[candidate_index, 0]
                    - prototype_logits[candidate_index, 1]
                ),
            }
        )
    return rows


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
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


def _best_balanced_accuracy_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(scores[np.isfinite(scores)])
    if not len(thresholds) or labels.all() or (~labels).all():
        return float("nan"), float("nan")
    best = (-1.0, float(thresholds[-1]))
    for threshold in thresholds:
        predicted = scores >= threshold
        sensitivity = float(predicted[labels].mean())
        specificity = float((~predicted[~labels]).mean())
        balanced = 0.5 * (sensitivity + specificity)
        candidate = (balanced, float(threshold))
        if candidate > best:
            best = candidate
    return best[1], best[0]


def fit_candidate_joint_calibration(
    rows: List[Dict[str, object]], beta_grid: Iterable[float]
) -> Dict[str, object]:
    selected = [
        row
        for row in rows
        if row["category"] in ("tp", "background_far_fp")
    ]
    labels = np.asarray([row["category"] == "tp" for row in selected])
    if not len(selected) or labels.all() or (~labels).all():
        raise ValueError("source calibration needs TP and background-far FP candidates")
    raw = np.asarray([row["raw_margin"] for row in selected], dtype=np.float64)
    prototype = np.asarray(
        [row["prototype_margin"] for row in selected], dtype=np.float64
    )
    raw_mean, prototype_mean = float(raw.mean()), float(prototype.mean())
    raw_std, prototype_std = float(raw.std()), float(prototype.std())
    raw_std = raw_std if raw_std > 1e-12 else 1.0
    prototype_std = prototype_std if prototype_std > 1e-12 else 1.0
    raw_z = (raw - raw_mean) / raw_std
    prototype_z = (prototype - prototype_mean) / prototype_std
    candidates = []
    for beta in sorted(set(float(value) for value in beta_grid)):
        if beta < 0 or beta > 1:
            raise ValueError("candidate joint beta must be in [0, 1]")
        joint = (1.0 - beta) * raw_z + beta * prototype_z
        candidates.append((_binary_roc_auc(labels, joint), -beta, beta, joint))
    if not candidates:
        raise ValueError("candidate joint beta grid is empty")
    source_auc, _, beta, joint = max(candidates, key=lambda row: (row[0], row[1]))
    threshold, balanced_accuracy = _best_balanced_accuracy_threshold(labels, joint)
    return {
        "version": CANDIDATE_AUDIT_VERSION,
        "beta": float(beta),
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "prototype_mean": prototype_mean,
        "prototype_std": prototype_std,
        "joint_threshold": threshold,
        "source_auc": float(source_auc),
        "source_balanced_accuracy": balanced_accuracy,
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
    }


def apply_candidate_joint_calibration(
    rows: List[Dict[str, object]], calibration: Dict[str, object]
) -> np.ndarray:
    if calibration.get("version") != CANDIDATE_AUDIT_VERSION:
        raise RuntimeError("candidate calibration version mismatch")
    raw = np.asarray([row["raw_margin"] for row in rows], dtype=np.float64)
    prototype = np.asarray(
        [row["prototype_margin"] for row in rows], dtype=np.float64
    )
    raw_z = (raw - float(calibration["raw_mean"])) / float(calibration["raw_std"])
    prototype_z = (
        prototype - float(calibration["prototype_mean"])
    ) / float(calibration["prototype_std"])
    beta = float(calibration["beta"])
    return (1.0 - beta) * raw_z + beta * prototype_z


def candidate_separation_report(
    rows: List[Dict[str, object]],
    score_names: Iterable[str] = ("raw_margin", "prototype_margin", "joint_score"),
) -> List[Dict[str, object]]:
    comparisons = {
        "tp_vs_background_far_fp": {"background_far_fp"},
        "tp_vs_near_miss_fp": {"near_miss_fp"},
        "tp_vs_all_fp": {
            "duplicate_fp",
            "near_miss_fp",
            "background_far_fp",
            "empty_image_fp",
        },
    }
    report = []
    for comparison, negatives in comparisons.items():
        selected = [
            row for row in rows if row["category"] == "tp" or row["category"] in negatives
        ]
        labels = np.asarray([row["category"] == "tp" for row in selected])
        for score_name in score_names:
            scores = np.asarray(
                [row[score_name] for row in selected], dtype=np.float64
            )
            slide_aucs = []
            slide_groups = sorted(set(str(row.get("slide_group", "")) for row in selected))
            for slide_group in slide_groups:
                mask = np.asarray(
                    [str(row.get("slide_group", "")) == slide_group for row in selected]
                )
                value = _binary_roc_auc(labels[mask], scores[mask])
                if np.isfinite(value):
                    slide_aucs.append(value)
            report.append(
                {
                    "comparison": comparison,
                    "score": score_name,
                    "global_auc": _binary_roc_auc(labels, scores),
                    "macro_slide_auc": (
                        float(np.mean(slide_aucs)) if slide_aucs else float("nan")
                    ),
                    "min_slide_auc": (
                        float(np.min(slide_aucs)) if slide_aucs else float("nan")
                    ),
                    "valid_slides": len(slide_aucs),
                    "positive_count": int(labels.sum()),
                    "negative_count": int((~labels).sum()),
                }
            )
    return report


def empty_counts() -> Dict[str, float]:
    return {key: 0.0 for key in COUNT_KEYS}


def evaluate_image(
    points: np.ndarray,
    logits: np.ndarray,
    gt_points: np.ndarray,
    dedup_interval: float,
    match_dis: float,
    near_radius: float,
) -> Dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    if len(points) != len(logits) or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("points and binary logits are misaligned")

    probabilities = softmax(logits)
    classes = np.argmax(probabilities, axis=-1)
    reserved = classes == 0
    pred_points, pred_scores = deduplicate_scores(
        points[reserved], probabilities[reserved], dedup_interval
    )
    cell_scores = pred_scores[:, 0] if len(pred_scores) else np.zeros((0,))
    matches, matched_pred, matched_gt = greedy_match(
        pred_points, cell_scores, gt_points, match_dis
    )

    result = empty_counts()
    result.update(
        images=1.0,
        gt=float(len(gt_points)),
        pred=float(len(pred_points)),
        tp=float(len(matches)),
        fp=float(len(pred_points) - len(matches)),
        fn=float(len(gt_points) - len(matches)),
    )

    for gt_index, gt in enumerate(gt_points):
        if gt_index in matched_gt:
            continue
        nearest_reserved = None
        if len(pred_points):
            nearest_reserved = float(distance_matrix(pred_points, gt[None]).min())
        nearest_raw = None
        nearest_raw_class = 1
        if len(points):
            raw_distances = distance_matrix(points, gt[None])[:, 0]
            raw_index = int(np.argmin(raw_distances))
            nearest_raw = float(raw_distances[raw_index])
            nearest_raw_class = int(classes[raw_index])
        if nearest_reserved is not None and match_dis < nearest_reserved <= near_radius:
            result["fn_localization_shift"] += 1
        elif nearest_raw is not None and nearest_raw <= near_radius:
            key = (
                "fn_low_score_or_background"
                if nearest_raw_class == 1
                else "fn_near_candidate_unmatched"
            )
            result[key] += 1
        else:
            result["fn_no_candidate"] += 1

    for pred_index, pred in enumerate(pred_points):
        if pred_index in matched_pred:
            continue
        if not len(gt_points):
            result["fp_empty_image"] += 1
            continue
        distances = distance_matrix(pred[None], gt_points)[0]
        nearest_gt = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_gt])
        if nearest_distance <= match_dis and nearest_gt in matched_gt:
            result["fp_duplicate"] += 1
        elif nearest_distance <= near_radius:
            result["fp_near_miss"] += 1
        else:
            result["fp_background_far"] += 1
    return result


def merge_counts(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key in COUNT_KEYS:
        target[key] += float(source.get(key, 0.0))


def summarize_counts(counts: Dict[str, float]) -> Dict[str, float]:
    tp = float(counts["tp"])
    pred = float(counts["pred"])
    gt = float(counts["gt"])
    precision = tp / pred if pred else 0.0
    recall = tp / gt if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **{key: float(counts[key]) for key in COUNT_KEYS},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_gt_ratio": pred / gt if gt else 0.0,
    }


def decide_rescoring(
    rows: Iterable[Dict[str, float]],
    minimum_f1_gain: float = 0.003,
    max_cls_fn_increase_fraction: float = 0.02,
) -> Dict[str, float]:
    rows = [dict(row) for row in rows]
    baseline_rows = [row for row in rows if float(row["alpha"]) == 0.0]
    candidates = [row for row in rows if float(row["alpha"]) > 0.0]
    if len(baseline_rows) != 1 or not candidates:
        raise ValueError("rescoring decision needs one alpha=0 row and non-zero candidates")
    baseline = baseline_rows[0]
    best = max(candidates, key=lambda row: float(row["f1"]))
    f1_gain = float(best["f1"]) - float(baseline["f1"])
    bg_far_delta = float(best["fp_background_far"]) - float(
        baseline["fp_background_far"]
    )
    cls_fn_delta = float(best["fn_low_score_or_background"]) - float(
        baseline["fn_low_score_or_background"]
    )
    cls_fn_limit = max_cls_fn_increase_fraction * float(baseline["gt"])
    if (
        f1_gain >= minimum_f1_gain
        and bg_far_delta <= 0
        and cls_fn_delta <= cls_fn_limit
    ):
        status = "promising"
        action = "validate_fixed_alpha_on_full_target_without_retuning"
    elif f1_gain > 0:
        status = "tradeoff_only"
        action = "inspect_fp_fn_tradeoff_before_training"
    else:
        status = "no_detection_gain"
        action = "do_not_integrate_current_frozen_bank"
    return {
        "status": status,
        "action": action,
        "diagnostic_only": True,
        "baseline_alpha": 0.0,
        "best_alpha": float(best["alpha"]),
        "baseline_f1": float(baseline["f1"]),
        "best_f1": float(best["f1"]),
        "f1_gain": f1_gain,
        "bg_far_fp_delta": bg_far_delta,
        "cls_fn_delta": cls_fn_delta,
        "cls_fn_increase_limit": cls_fn_limit,
    }
