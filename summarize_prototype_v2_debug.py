import argparse
import json
import math
import os
import re
import statistics
from typing import Dict, Iterable, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Prototype-v2 initialization, sampling, EMA, and gradients."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def collect(rows: Iterable[Dict], section: str, key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(section, {}).get(key)
        if finite_number(value):
            values.append(float(value))
    return values


def mean_or_none(values: List[float]):
    return statistics.mean(values) if values else None


def last_or_none(values: List[float]):
    return values[-1] if values else None


def first_or_none(values: List[float]):
    return values[0] if values else None


def assignment_shares(rows: Iterable[Dict], key: str) -> Tuple[List[float], List[float]]:
    """Return per-epoch maximum and minimum non-empty prototype occupancy."""
    maximums: List[float] = []
    minimums: List[float] = []
    for row in rows:
        counts = row.get("prototype_state", {}).get(key)
        if not isinstance(counts, list) or not counts:
            continue
        total = sum(float(value) for value in counts)
        if total <= 0:
            continue
        shares = [float(value) / total for value in counts]
        maximums.append(max(shares))
        minimums.append(min(shares))
    return maximums, minimums


def indexed_step_values(step_mean: Dict, prefix: str) -> Dict[int, float]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_count$")
    values: Dict[int, float] = {}
    for key, value in step_mean.items():
        match = pattern.match(key)
        if match and finite_number(value):
            values[int(match.group(1))] = float(value)
    return values


def format_value(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def audit(rows: List[Dict]) -> Tuple[List[str], List[str], Dict]:
    failures: List[str] = []
    warnings: List[str] = []
    if not rows:
        return ["prototype_debug.jsonl is empty"], warnings, {}

    initialized_epochs = [
        int(row["epoch"])
        for row in rows
        if int(row.get("prototype_state", {}).get("prototype_initialized_now", 0)) == 1
    ]
    ready_epochs = [
        int(row["epoch"])
        for row in rows
        if int(row.get("prototype_state", {}).get("prototype_ready", 0)) == 1
    ]
    if len(initialized_epochs) != 1:
        failures.append(
            f"expected exactly one KMeans initialization, observed {initialized_epochs}"
        )
    if not ready_epochs:
        failures.append("prototype bank never became ready")

    positive = collect(rows, "step_mean", "prototype_positive")
    rejected = collect(rows, "step_mean", "prototype_rejected_positive")
    hard_bg = collect(rows, "step_mean", "prototype_hard_background")
    random_bg = collect(rows, "step_mean", "prototype_random_background")
    losses = collect(rows, "step_mean", "prototype_loss_raw")
    positive_losses = collect(rows, "step_mean", "prototype_loss_positive")
    hard_background_losses = collect(
        rows, "step_mean", "prototype_loss_hard_background"
    )
    random_background_losses = collect(
        rows, "step_mean", "prototype_loss_random_background"
    )
    gradient_ratio = collect(rows, "step_mean", "gradient_all_proto_cls_ratio")
    gradient_cosine = collect(rows, "step_mean", "gradient_all_cosine")
    shared_gradient_available = collect(
        rows, "step_mean", "gradient_cls_features_available"
    )
    fg_drift = collect(rows, "prototype_state", "foreground_center_drift_cosine")
    bg_drift = collect(rows, "prototype_state", "background_center_drift_cosine")
    cross_mean = collect(
        rows, "prototype_state", "foreground_background_similarity_mean"
    )
    cross_max = collect(
        rows, "prototype_state", "foreground_background_similarity_max"
    )
    fresh_cross_mean = collect(
        rows, "prototype_state", "fresh_foreground_background_similarity_mean"
    )
    fresh_cross_max = collect(
        rows, "prototype_state", "fresh_foreground_background_similarity_max"
    )
    fg_fresh_old = collect(
        rows, "prototype_state", "foreground_fresh_old_alignment_cosine"
    )
    bg_fresh_old = collect(
        rows, "prototype_state", "background_fresh_old_alignment_cosine"
    )
    fg_updated_fresh = collect(
        rows, "prototype_state", "foreground_updated_fresh_cosine"
    )
    bg_updated_fresh = collect(
        rows, "prototype_state", "background_updated_fresh_cosine"
    )
    projector_probe_drift = collect(
        rows, "prototype_state", "projector_probe_drift_cosine"
    )
    positive_margin = collect(rows, "step_mean", "prototype_positive_margin")
    background_margin = collect(rows, "step_mean", "prototype_background_margin")
    positive_correct = collect(
        rows, "step_mean", "prototype_positive_correct_rate"
    )
    background_correct = collect(
        rows, "step_mean", "prototype_background_correct_rate"
    )
    hard_background_correct = collect(
        rows, "step_mean", "prototype_hard_background_correct_rate"
    )
    random_background_correct = collect(
        rows, "step_mean", "prototype_random_background_correct_rate"
    )
    training_batch_fraction = collect(rows, "step_mean", "train_batch_fraction")
    periodic_refresh_enabled = collect(
        rows, "prototype_state", "prototype_periodic_refresh_enabled"
    )
    refresh_events = collect(
        rows, "prototype_state", "prototype_epoch_refresh_events"
    )
    refresh_post_positive = collect(
        rows,
        "prototype_state",
        "prototype_epoch_refresh_post_positive_correct_rate",
    )
    refresh_post_background = collect(
        rows,
        "prototype_state",
        "prototype_epoch_refresh_post_background_correct_rate",
    )
    refresh_projector_drift_min = collect(
        rows,
        "prototype_state",
        "prototype_epoch_refresh_projector_drift_min",
    )
    refresh_center_age_max = collect(
        rows, "prototype_state", "prototype_epoch_refresh_center_age_max"
    )
    refresh_reprojected_rate = collect(
        rows, "prototype_state", "prototype_epoch_refresh_reprojected_rate"
    )

    if not positive or max(positive) <= 0:
        failures.append("no accepted positive samples were observed")
    if not hard_bg or max(hard_bg) <= 0:
        failures.append("no hard-background samples were observed")
    if not random_bg or max(random_bg) <= 0:
        warnings.append(
            "random-background sampling is disabled; the prototype bank is "
            "being trained on hard background only"
        )
    if losses and not all(finite_number(value) for value in losses):
        failures.append("prototype loss contains a non-finite value")
    if ready_epochs and not gradient_ratio:
        failures.append("prototype/classification gradient diagnostics are missing")
    mean_shared_gradient_available = mean_or_none(shared_gradient_available)
    if (
        mean_shared_gradient_available is not None
        and mean_shared_gradient_available < 0.999
    ):
        failures.append(
            "shared-feature gradient diagnostics are unavailable; do not interpret "
            "the serialized gradient ratio as a true zero"
        )
    mean_training_batch_fraction = mean_or_none(training_batch_fraction)
    if (
        mean_training_batch_fraction is not None
        and mean_training_batch_fraction < 0.999
    ):
        warnings.append(
            "prototype centers were updated from truncated epochs "
            f"(mean batch fraction {mean_training_batch_fraction:.4f}); "
            "use this run only as a rapid mechanism gate"
        )

    mean_gradient_ratio = mean_or_none(gradient_ratio)
    mean_gradient_cosine = mean_or_none(gradient_cosine)
    if mean_gradient_ratio is not None and mean_gradient_ratio > 0.30:
        warnings.append(
            f"prototype/classification gradient norm ratio is high ({mean_gradient_ratio:.4f})"
        )
    if mean_gradient_cosine is not None and mean_gradient_cosine < -0.10:
        warnings.append(
            f"prototype/classification gradients are strongly conflicting ({mean_gradient_cosine:.4f})"
        )

    post_init_rows = [
        row
        for row in rows
        if initialized_epochs and int(row["epoch"]) > initialized_epochs[0]
    ]
    if len(post_init_rows) < 3:
        warnings.append(
            f"only {len(post_init_rows)} post-initialization epoch(s) were observed; "
            "prototype separation and EMA stability are not yet established"
        )
    post_fg_drift = collect(
        post_init_rows, "prototype_state", "foreground_center_drift_cosine"
    )
    post_bg_drift = collect(
        post_init_rows, "prototype_state", "background_center_drift_cosine"
    )
    if post_fg_drift and min(post_fg_drift) < 0.90:
        warnings.append(
            f"foreground prototype update is unstable (minimum drift cosine {min(post_fg_drift):.4f})"
        )
    if post_bg_drift and min(post_bg_drift) < 0.90:
        warnings.append(
            f"background prototype update is unstable (minimum drift cosine {min(post_bg_drift):.4f})"
        )
    mean_cross = mean_or_none(cross_mean)
    if mean_cross is not None and mean_cross > 0.80:
        warnings.append(
            f"foreground/background centers are weakly separated (mean cosine {mean_cross:.4f})"
        )
    maximum_cross = max(cross_max) if cross_max else None
    final_cross_mean = last_or_none(cross_mean)
    final_cross_max = last_or_none(cross_max)
    final_fresh_cross_mean = last_or_none(fresh_cross_mean)
    final_fresh_cross_max = last_or_none(fresh_cross_max)
    final_state = rows[-1].get("prototype_state", {})
    final_step = rows[-1].get("step_mean", {})
    worst_fg_index = final_state.get("worst_cross_foreground_index")
    worst_bg_index = final_state.get("worst_cross_background_index")
    worst_fg_share = final_state.get("worst_cross_foreground_assignment_share")
    worst_bg_share = final_state.get("worst_cross_background_assignment_share")
    hard_center_counts = indexed_step_values(
        final_step, "prototype_hard_background_bg_center_"
    )
    random_center_counts = indexed_step_values(
        final_step, "prototype_random_background_bg_center_"
    )
    background_center_source = []
    for center_idx in sorted(set(hard_center_counts) | set(random_center_counts)):
        hard_count = hard_center_counts.get(center_idx, 0.0)
        random_count = random_center_counts.get(center_idx, 0.0)
        total = hard_count + random_count
        background_center_source.append(
            {
                "center": center_idx,
                "hard_count_per_step": hard_count,
                "random_count_per_step": random_count,
                "hard_fraction": hard_count / total if total > 0 else None,
            }
        )
    worst_bg_source = next(
        (
            item
            for item in background_center_source
            if item["center"] == worst_bg_index
        ),
        None,
    )
    if final_cross_max is not None and final_cross_max > 0.95:
        pair_usage = (
            finite_number(worst_fg_share)
            and finite_number(worst_bg_share)
            and min(float(worst_fg_share), float(worst_bg_share)) >= 0.05
        )
        usage_note = "high-usage collision" if pair_usage else "low-usage collision"
        warnings.append(
            f"foreground/background prototype pair fg={worst_fg_index}, bg={worst_bg_index} "
            f"nearly overlaps in the final epoch (cosine {final_cross_max:.4f}, "
            f"shares {worst_fg_share}/{worst_bg_share}, {usage_note})"
        )
    if final_fresh_cross_mean is not None and final_fresh_cross_mean > 0.75:
        warnings.append(
            f"fresh foreground/background KMeans targets overlap "
            f"(mean cosine {final_fresh_cross_mean:.4f})"
        )
    post_projector_drift = projector_probe_drift[1:] if len(projector_probe_drift) > 1 else []
    if post_projector_drift and min(post_projector_drift) < 0.90:
        warnings.append(
            f"prototype projector space is unstable "
            f"(minimum fixed-probe cosine {min(post_projector_drift):.4f})"
        )

    post_fg_max, post_fg_min = assignment_shares(
        post_init_rows, "foreground_assignment_counts"
    )
    post_bg_max, post_bg_min = assignment_shares(
        post_init_rows, "background_assignment_counts"
    )
    final_fg_max = last_or_none(post_fg_max)
    final_fg_min = last_or_none(post_fg_min)
    final_bg_max = last_or_none(post_bg_max)
    final_bg_min = last_or_none(post_bg_min)
    if final_fg_max is not None and final_fg_max > 0.60:
        warnings.append(
            f"foreground prototype occupancy is concentrated "
            f"in the final epoch (maximum share {final_fg_max:.4f})"
        )
    if final_bg_max is not None and final_bg_max > 0.50:
        warnings.append(
            f"background prototype occupancy is concentrated "
            f"in the final epoch (maximum share {final_bg_max:.4f})"
        )
    if final_fg_min is not None and final_fg_min < 0.01:
        warnings.append(
            f"a foreground prototype is nearly unused "
            f"in the final epoch (minimum share {final_fg_min:.4f})"
        )
    if final_bg_min is not None and final_bg_min < 0.01:
        warnings.append(
            f"a background prototype is nearly unused "
            f"in the final epoch (minimum share {final_bg_min:.4f})"
        )

    final_positive_margin = last_or_none(positive_margin)
    final_background_margin = last_or_none(background_margin)
    initial_hard_background_correct = first_or_none(hard_background_correct)
    final_hard_background_correct = last_or_none(hard_background_correct)
    hard_background_correct_delta = (
        final_hard_background_correct - initial_hard_background_correct
        if final_hard_background_correct is not None
        and initial_hard_background_correct is not None
        else None
    )
    if final_positive_margin is not None and final_positive_margin < 0.50:
        warnings.append(
            f"foreground prototype margin is weak ({final_positive_margin:.4f})"
        )
    if final_background_margin is not None and final_background_margin > -0.50:
        warnings.append(
            f"background prototype margin is weak ({final_background_margin:.4f})"
        )
    if mean_gradient_ratio is not None and mean_gradient_ratio < 0.02:
        warnings.append(
            f"prototype gradient reaching shared P2P features is weak "
            f"({mean_gradient_ratio:.4f} of classification gradient)"
        )

    final_fg_counts = final_state.get("foreground_assignment_counts", [])
    final_bg_counts = final_state.get("background_assignment_counts", [])
    fg_effective = final_state.get("foreground_effective_prototypes")
    bg_effective = final_state.get("background_effective_prototypes")
    periodic_refresh_active = bool(
        periodic_refresh_enabled and periodic_refresh_enabled[-1] >= 0.5
    )
    mechanism_gates = {
        "shared_gradient_diagnostics_available": (
            mean_shared_gradient_available is None
            or mean_shared_gradient_available >= 0.999
        ),
        "positive_correct_at_least_0p75": (
            last_or_none(positive_correct) is not None
            and last_or_none(positive_correct) >= 0.75
        ),
        "hard_background_correct_at_least_0p50": (
            final_hard_background_correct is not None
            and final_hard_background_correct >= 0.50
        ),
        "hard_background_drop_no_more_than_0p10": (
            hard_background_correct_delta is not None
            and hard_background_correct_delta >= -0.10
        ),
        "foreground_effective_at_least_70pct": (
            finite_number(fg_effective)
            and len(final_fg_counts) > 0
            and float(fg_effective) >= 0.70 * len(final_fg_counts)
        ),
        "background_effective_at_least_70pct": (
            finite_number(bg_effective)
            and len(final_bg_counts) > 0
            and float(bg_effective) >= 0.70 * len(final_bg_counts)
        ),
        "cross_similarity_mean_at_most_0p75": (
            final_cross_mean is not None and final_cross_mean <= 0.75
        ),
        "fresh_cross_similarity_mean_at_most_0p75": (
            final_fresh_cross_mean is not None and final_fresh_cross_mean <= 0.75
        ),
        "projector_probe_drift_at_least_0p90": (
            bool(post_projector_drift) and min(post_projector_drift) >= 0.90
        ),
        "ema_targets_tracked_at_least_0p90": (
            bool(fg_updated_fresh)
            and bool(bg_updated_fresh)
            and last_or_none(fg_updated_fresh) >= 0.90
            and last_or_none(bg_updated_fresh) >= 0.90
        ),
        "gradient_ratio_in_0p02_to_0p30": (
            mean_gradient_ratio is not None
            and 0.02 <= mean_gradient_ratio <= 0.30
        ),
        "gradient_cosine_at_least_minus_0p10": (
            mean_gradient_cosine is not None and mean_gradient_cosine >= -0.10
        ),
    }
    if periodic_refresh_active:
        refresh_interval = final_state.get("prototype_refresh_interval_steps")
        final_refresh_age = last_or_none(refresh_center_age_max)
        mechanism_gates.update(
            {
                "periodic_refresh_events_present": (
                    bool(refresh_events) and min(refresh_events) > 0
                ),
                "refresh_positive_correct_at_least_0p75": (
                    last_or_none(refresh_post_positive) is not None
                    and last_or_none(refresh_post_positive) >= 0.75
                ),
                "refresh_background_correct_at_least_0p50": (
                    last_or_none(refresh_post_background) is not None
                    and last_or_none(refresh_post_background) >= 0.50
                ),
                "refresh_projector_drift_at_least_0p97": (
                    bool(refresh_projector_drift_min)
                    and min(refresh_projector_drift_min) >= 0.97
                ),
                "refresh_center_age_within_interval": (
                    finite_number(final_refresh_age)
                    and finite_number(refresh_interval)
                    and float(final_refresh_age) <= float(refresh_interval)
                ),
                "refresh_uses_current_projector": (
                    bool(refresh_reprojected_rate)
                    and min(refresh_reprojected_rate) >= 0.999
                ),
            }
        )
    long_run_ready = not failures and all(mechanism_gates.values())
    if not long_run_ready:
        failed_gates = [key for key, passed in mechanism_gates.items() if not passed]
        warnings.append(
            "not ready for a long run; failed mechanism gates: "
            + ", ".join(failed_gates)
        )

    summary = {
        "epochs_logged": [int(row["epoch"]) for row in rows],
        "initialized_epochs": initialized_epochs,
        "ready_epochs": ready_epochs,
        "mean_positive_per_step": mean_or_none(positive),
        "mean_rejected_positive_per_step": mean_or_none(rejected),
        "mean_hard_background_per_step": mean_or_none(hard_bg),
        "mean_random_background_per_step": mean_or_none(random_bg),
        "mean_training_batch_fraction": mean_training_batch_fraction,
        "mean_prototype_loss": mean_or_none(losses),
        "mean_positive_prototype_loss": mean_or_none(positive_losses),
        "mean_hard_background_prototype_loss": mean_or_none(
            hard_background_losses
        ),
        "mean_random_background_prototype_loss": mean_or_none(
            random_background_losses
        ),
        "mean_gradient_proto_cls_ratio": mean_gradient_ratio,
        "mean_gradient_cosine": mean_gradient_cosine,
        "mean_shared_gradient_available": mean_shared_gradient_available,
        "minimum_foreground_drift_cosine_after_init": (
            min(post_fg_drift) if post_fg_drift else None
        ),
        "minimum_background_drift_cosine_after_init": (
            min(post_bg_drift) if post_bg_drift else None
        ),
        "mean_foreground_background_similarity": mean_cross,
        "max_foreground_background_similarity": maximum_cross,
        "final_foreground_background_similarity_mean": final_cross_mean,
        "final_foreground_background_similarity_max": final_cross_max,
        "final_fresh_foreground_background_similarity_mean": final_fresh_cross_mean,
        "final_fresh_foreground_background_similarity_max": final_fresh_cross_max,
        "minimum_projector_probe_drift_cosine_after_init": (
            min(post_projector_drift) if post_projector_drift else None
        ),
        "final_foreground_fresh_old_alignment_cosine": last_or_none(fg_fresh_old),
        "final_background_fresh_old_alignment_cosine": last_or_none(bg_fresh_old),
        "final_foreground_updated_fresh_cosine": last_or_none(fg_updated_fresh),
        "final_background_updated_fresh_cosine": last_or_none(bg_updated_fresh),
        "final_background_center_source_composition": background_center_source,
        "final_worst_cross_background_source_composition": worst_bg_source,
        "final_positive_margin": final_positive_margin,
        "final_background_margin": final_background_margin,
        "final_positive_correct_rate": last_or_none(positive_correct),
        "final_background_correct_rate": last_or_none(background_correct),
        "initial_hard_background_correct_rate": initial_hard_background_correct,
        "final_hard_background_correct_rate": final_hard_background_correct,
        "hard_background_correct_rate_delta": hard_background_correct_delta,
        "final_random_background_correct_rate": last_or_none(random_background_correct),
        "periodic_refresh_active": periodic_refresh_active,
        "final_refresh_events": last_or_none(refresh_events),
        "final_refresh_post_positive_correct_rate": last_or_none(
            refresh_post_positive
        ),
        "final_refresh_post_background_correct_rate": last_or_none(
            refresh_post_background
        ),
        "minimum_refresh_projector_drift_cosine": (
            min(refresh_projector_drift_min)
            if refresh_projector_drift_min else None
        ),
        "maximum_refresh_center_age_steps": (
            max(refresh_center_age_max) if refresh_center_age_max else None
        ),
        "minimum_refresh_reprojected_rate": (
            min(refresh_reprojected_rate) if refresh_reprojected_rate else None
        ),
        "final_worst_cross_foreground_index": worst_fg_index,
        "final_worst_cross_background_index": worst_bg_index,
        "final_worst_cross_foreground_share": worst_fg_share,
        "final_worst_cross_background_share": worst_bg_share,
        "final_foreground_effective_prototypes": final_state.get(
            "foreground_effective_prototypes"
        ),
        "final_background_effective_prototypes": final_state.get(
            "background_effective_prototypes"
        ),
        "maximum_foreground_assignment_share": (
            max(post_fg_max) if post_fg_max else None
        ),
        "minimum_foreground_assignment_share": (
            min(post_fg_min) if post_fg_min else None
        ),
        "maximum_background_assignment_share": (
            max(post_bg_max) if post_bg_max else None
        ),
        "minimum_background_assignment_share": (
            min(post_bg_min) if post_bg_min else None
        ),
        "final_foreground_assignment_max_share": final_fg_max,
        "final_foreground_assignment_min_share": final_fg_min,
        "final_background_assignment_max_share": final_bg_max,
        "final_background_assignment_min_share": final_bg_min,
        "post_initialization_epochs": len(post_init_rows),
        "mechanism_gates": mechanism_gates,
        "long_run_ready": long_run_ready,
        "failures": failures,
        "warnings": warnings,
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
    }
    return failures, warnings, summary


def main():
    args = parse_args()
    debug_path = os.path.join(args.experiment, "prototype_debug.jsonl")
    if not os.path.isfile(debug_path):
        raise FileNotFoundError(f"prototype debug log not found: {debug_path}")
    with open(debug_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    failures, warnings, summary = audit(rows)
    output = args.output or os.path.join(args.experiment, "prototype_v2_audit.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"status={summary['status']}")
    print(f"epochs={summary['epochs_logged']}")
    print(f"KMeans initialization epochs={summary['initialized_epochs']}")
    print(f"prototype ready epochs={summary['ready_epochs']}")
    for key in (
        "mean_positive_per_step",
        "mean_rejected_positive_per_step",
        "mean_hard_background_per_step",
        "mean_random_background_per_step",
        "mean_prototype_loss",
        "mean_positive_prototype_loss",
        "mean_hard_background_prototype_loss",
        "mean_random_background_prototype_loss",
        "mean_gradient_proto_cls_ratio",
        "mean_gradient_cosine",
        "mean_shared_gradient_available",
        "minimum_foreground_drift_cosine_after_init",
        "minimum_background_drift_cosine_after_init",
        "mean_foreground_background_similarity",
        "max_foreground_background_similarity",
        "final_foreground_background_similarity_mean",
        "final_foreground_background_similarity_max",
        "final_fresh_foreground_background_similarity_mean",
        "final_fresh_foreground_background_similarity_max",
        "minimum_projector_probe_drift_cosine_after_init",
        "final_foreground_fresh_old_alignment_cosine",
        "final_background_fresh_old_alignment_cosine",
        "final_foreground_updated_fresh_cosine",
        "final_background_updated_fresh_cosine",
        "final_positive_margin",
        "final_background_margin",
        "final_positive_correct_rate",
        "final_background_correct_rate",
        "initial_hard_background_correct_rate",
        "final_hard_background_correct_rate",
        "hard_background_correct_rate_delta",
        "final_random_background_correct_rate",
        "periodic_refresh_active",
        "final_refresh_events",
        "final_refresh_post_positive_correct_rate",
        "final_refresh_post_background_correct_rate",
        "minimum_refresh_projector_drift_cosine",
        "maximum_refresh_center_age_steps",
        "minimum_refresh_reprojected_rate",
        "final_worst_cross_foreground_index",
        "final_worst_cross_background_index",
        "final_worst_cross_foreground_share",
        "final_worst_cross_background_share",
        "final_foreground_effective_prototypes",
        "final_background_effective_prototypes",
        "maximum_foreground_assignment_share",
        "minimum_foreground_assignment_share",
        "maximum_background_assignment_share",
        "minimum_background_assignment_share",
        "final_foreground_assignment_max_share",
        "final_foreground_assignment_min_share",
        "final_background_assignment_max_share",
        "final_background_assignment_min_share",
        "post_initialization_epochs",
        "long_run_ready",
    ):
        print(f"{key}={format_value(summary[key])}")
    for message in failures:
        print(f"FAIL: {message}")
    for message in warnings:
        print(f"WARN: {message}")
    print(f"audit_json={output}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
