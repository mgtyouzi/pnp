import argparse
import csv
import json
import math
import os
from argparse import Namespace
from collections import defaultdict

import numpy as np


PROTOTYPE_GUIDED_ASSIGNMENT_AUDIT_VERSION = (
    "prototype_guided_assignment_audit_v2_20260907"
)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_lambdas(specification):
    values = sorted(
        {
            float(item.strip())
            for item in str(specification).split(",")
            if item.strip()
        }
    )
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("prototype weights must be finite positive values")
    return values


def build_guided_cost(
    baseline_cost,
    point_distances,
    prototype_similarities,
    prototype_weight,
    local_radius,
):
    baseline = np.asarray(baseline_cost, dtype=np.float64)
    distances = np.asarray(point_distances, dtype=np.float64)
    similarities = np.asarray(prototype_similarities, dtype=np.float64)
    if baseline.shape != distances.shape or baseline.shape != similarities.shape:
        raise ValueError("cost, distance, and similarity matrices must have equal shape")
    if prototype_weight <= 0 or local_radius <= 0:
        raise ValueError("prototype_weight and local_radius must be positive")
    local_penalty = np.where(
        distances <= float(local_radius),
        1.0 - np.clip(similarities, -1.0, 1.0),
        1.0,
    )
    return baseline + float(prototype_weight) * local_penalty


def solve_target_assignment(cost):
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("assignment cost must be a two-dimensional matrix")
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=np.int64)
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError("candidate count must be at least target count")
    try:
        from scipy.optimize import linear_sum_assignment

        candidate_indices, target_indices = linear_sum_assignment(matrix)
    except ModuleNotFoundError:
        if matrix.shape[0] > 8 or matrix.shape[1] > 8:
            raise RuntimeError(
                "SciPy is required for production-size Hungarian assignment"
            )
        from itertools import permutations

        target_indices = np.arange(matrix.shape[1], dtype=np.int64)
        candidate_indices = np.asarray(
            min(
                permutations(range(matrix.shape[0]), matrix.shape[1]),
                key=lambda candidates: matrix[
                    np.asarray(candidates, dtype=np.int64), target_indices
                ].sum(),
            ),
            dtype=np.int64,
        )
    target_to_candidate = np.full(matrix.shape[1], -1, dtype=np.int64)
    target_to_candidate[target_indices] = candidate_indices
    if np.any(target_to_candidate < 0):
        raise RuntimeError("Hungarian assignment did not cover every target")
    return target_to_candidate


def build_target_switch_records(
    baseline_assignment,
    guided_assignment,
    candidate_scores,
    point_distances,
    prototype_similarities,
    match_radius=15.0,
    minimum_distance_improvement=3.0,
    minimum_similarity_improvement=0.1,
    minimum_candidate_score=0.2,
    maximum_score_drop=0.25,
):
    baseline = np.asarray(baseline_assignment, dtype=np.int64).reshape(-1)
    guided = np.asarray(guided_assignment, dtype=np.int64).reshape(-1)
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    distances = np.asarray(point_distances, dtype=np.float64)
    similarities = np.asarray(prototype_similarities, dtype=np.float64)
    if baseline.shape != guided.shape:
        raise ValueError("baseline and guided assignments must cover equal targets")
    if distances.shape != similarities.shape or distances.shape[1] != len(baseline):
        raise ValueError("distance and similarity matrices must align with targets")
    if len(scores) != distances.shape[0]:
        raise ValueError("candidate scores must align with matrix rows")

    records = []
    for target_index in np.flatnonzero(baseline != guided):
        baseline_candidate = int(baseline[target_index])
        guided_candidate = int(guided[target_index])
        baseline_score = float(scores[baseline_candidate])
        guided_score = float(scores[guided_candidate])
        baseline_distance = float(distances[baseline_candidate, target_index])
        guided_distance = float(distances[guided_candidate, target_index])
        baseline_similarity = float(
            similarities[baseline_candidate, target_index]
        )
        guided_similarity = float(similarities[guided_candidate, target_index])
        score_delta = guided_score - baseline_score
        distance_improvement = baseline_distance - guided_distance
        similarity_delta = guided_similarity - baseline_similarity
        is_local = guided_distance <= float(match_radius)
        improves_distance = distance_improvement >= float(
            minimum_distance_improvement
        )
        improves_similarity = similarity_delta >= float(
            minimum_similarity_improvement
        )
        passes_score_floor = guided_score >= float(minimum_candidate_score)
        passes_score_drop = score_delta >= -float(maximum_score_drop)
        reliable = (
            is_local
            and improves_distance
            and improves_similarity
            and passes_score_floor
            and passes_score_drop
        )
        records.append(
            {
                "target_index": int(target_index),
                "baseline_candidate": baseline_candidate,
                "guided_candidate": guided_candidate,
                "baseline_score": baseline_score,
                "guided_score": guided_score,
                "score_delta": score_delta,
                "baseline_distance": baseline_distance,
                "guided_distance": guided_distance,
                "distance_improvement": distance_improvement,
                "baseline_similarity": baseline_similarity,
                "guided_similarity": guided_similarity,
                "similarity_delta": similarity_delta,
                "is_local": bool(is_local),
                "improves_distance": bool(improves_distance),
                "improves_similarity": bool(improves_similarity),
                "passes_score_floor": bool(passes_score_floor),
                "passes_score_drop": bool(passes_score_drop),
                "reliable_switch": bool(reliable),
            }
        )
    return records


def compare_target_assignments(
    baseline_assignment,
    guided_assignment,
    candidate_scores,
    point_distances,
    prototype_similarities,
    score_threshold=0.5,
    match_radius=15.0,
    distance_tolerance=1.0,
    minimum_distance_improvement=3.0,
    minimum_similarity_improvement=0.1,
    minimum_candidate_score=0.2,
    maximum_score_drop=0.25,
):
    baseline = np.asarray(baseline_assignment, dtype=np.int64).reshape(-1)
    guided = np.asarray(guided_assignment, dtype=np.int64).reshape(-1)
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    distances = np.asarray(point_distances, dtype=np.float64)
    similarities = np.asarray(prototype_similarities, dtype=np.float64)
    if baseline.shape != guided.shape:
        raise ValueError("baseline and guided assignments must cover equal targets")
    if distances.shape != similarities.shape or distances.shape[1] != len(baseline):
        raise ValueError("distance and similarity matrices must align with targets")

    targets = np.arange(len(baseline), dtype=np.int64)
    baseline_scores = scores[baseline]
    guided_scores = scores[guided]
    baseline_distances = distances[baseline, targets]
    guided_distances = distances[guided, targets]
    baseline_similarities = similarities[baseline, targets]
    guided_similarities = similarities[guided, targets]
    switched = baseline != guided
    baseline_low = baseline_scores < float(score_threshold)
    low_to_high = switched & baseline_low & (guided_scores >= float(score_threshold))
    switch_records = build_target_switch_records(
        baseline,
        guided,
        scores,
        distances,
        similarities,
        match_radius=match_radius,
        minimum_distance_improvement=minimum_distance_improvement,
        minimum_similarity_improvement=minimum_similarity_improvement,
        minimum_candidate_score=minimum_candidate_score,
        maximum_score_drop=maximum_score_drop,
    )
    reliable_records = [
        record for record in switch_records if record["reliable_switch"]
    ]

    return {
        "images": 1,
        "targets": int(len(targets)),
        "switch_count": int(switched.sum()),
        "score_improved_switch_count": int(
            (switched & (guided_scores > baseline_scores)).sum()
        ),
        "local_switch_count": int(
            (switched & (guided_distances <= float(match_radius))).sum()
        ),
        "distance_safe_switch_count": int(
            (
                switched
                & (guided_distances <= baseline_distances + float(distance_tolerance))
            ).sum()
        ),
        "reliable_switch_count": len(reliable_records),
        "nonlocal_switch_count": sum(
            not record["is_local"] for record in switch_records
        ),
        "baseline_low_score_count": int(baseline_low.sum()),
        "low_to_high_count": int(low_to_high.sum()),
        "switched_score_delta_sum": float(
            (guided_scores[switched] - baseline_scores[switched]).sum()
        ),
        "switched_distance_delta_sum": float(
            (guided_distances[switched] - baseline_distances[switched]).sum()
        ),
        "switched_similarity_delta_sum": float(
            (guided_similarities[switched] - baseline_similarities[switched]).sum()
        ),
        "reliable_score_delta_sum": sum(
            record["score_delta"] for record in reliable_records
        ),
        "reliable_distance_improvement_sum": sum(
            record["distance_improvement"] for record in reliable_records
        ),
        "reliable_similarity_delta_sum": sum(
            record["similarity_delta"] for record in reliable_records
        ),
        "baseline_within_match_count": int(
            (baseline_distances <= float(match_radius)).sum()
        ),
        "guided_within_match_count": int(
            (guided_distances <= float(match_radius)).sum()
        ),
    }


def aggregate_assignment_metrics(per_image_metrics):
    grouped = defaultdict(list)
    for item in per_image_metrics:
        grouped[float(item["lambda"])].append(item)

    rows = []
    sum_fields = (
        "images",
        "targets",
        "switch_count",
        "score_improved_switch_count",
        "local_switch_count",
        "distance_safe_switch_count",
        "reliable_switch_count",
        "nonlocal_switch_count",
        "baseline_low_score_count",
        "low_to_high_count",
        "switched_score_delta_sum",
        "switched_distance_delta_sum",
        "switched_similarity_delta_sum",
        "reliable_score_delta_sum",
        "reliable_distance_improvement_sum",
        "reliable_similarity_delta_sum",
        "baseline_within_match_count",
        "guided_within_match_count",
    )
    for prototype_weight in sorted(grouped):
        totals = {
            field: sum(float(item.get(field, 0)) for item in grouped[prototype_weight])
            for field in sum_fields
        }
        switches = totals["switch_count"]
        targets = totals["targets"]
        row = {
            "lambda": float(prototype_weight),
            **{
                field: int(value) if field not in {
                    "switched_score_delta_sum",
                    "switched_distance_delta_sum",
                    "switched_similarity_delta_sum",
                    "reliable_score_delta_sum",
                    "reliable_distance_improvement_sum",
                    "reliable_similarity_delta_sum",
                } else float(value)
                for field, value in totals.items()
            },
            "switch_rate": safe_div(switches, targets),
            "score_improved_switch_rate": safe_div(
                totals["score_improved_switch_count"], switches
            ),
            "local_switch_rate": safe_div(totals["local_switch_count"], switches),
            "distance_safe_switch_rate": safe_div(
                totals["distance_safe_switch_count"], switches
            ),
            "reliable_switch_rate": safe_div(
                totals["reliable_switch_count"], targets
            ),
            "reliable_switch_share": safe_div(
                totals["reliable_switch_count"], switches
            ),
            "nonlocal_switch_rate": safe_div(
                totals["nonlocal_switch_count"], switches
            ),
            "low_to_high_rate": safe_div(
                totals["low_to_high_count"], totals["baseline_low_score_count"]
            ),
            "switched_score_delta_mean": safe_div(
                totals["switched_score_delta_sum"], switches
            ),
            "switched_distance_delta_mean": safe_div(
                totals["switched_distance_delta_sum"], switches
            ),
            "switched_similarity_delta_mean": safe_div(
                totals["switched_similarity_delta_sum"], switches
            ),
            "reliable_score_delta_mean": safe_div(
                totals["reliable_score_delta_sum"],
                totals["reliable_switch_count"],
            ),
            "reliable_distance_improvement_mean": safe_div(
                totals["reliable_distance_improvement_sum"],
                totals["reliable_switch_count"],
            ),
            "reliable_similarity_delta_mean": safe_div(
                totals["reliable_similarity_delta_sum"],
                totals["reliable_switch_count"],
            ),
            "baseline_within_match_rate": safe_div(
                totals["baseline_within_match_count"], targets
            ),
            "guided_within_match_rate": safe_div(
                totals["guided_within_match_count"], targets
            ),
            "within_match_rate_delta": safe_div(
                totals["guided_within_match_count"]
                - totals["baseline_within_match_count"],
                targets,
            ),
        }
        rows.append(row)
    return rows


def _checks(row, thresholds):
    return {
        "minimum_reliable_switch_count": row["reliable_switch_count"]
        >= thresholds["minimum_reliable_switch_count"],
        "minimum_reliable_switch_rate": row["reliable_switch_rate"]
        >= thresholds["minimum_reliable_switch_rate"],
        "maximum_reliable_switch_rate": row["reliable_switch_rate"]
        <= thresholds["maximum_reliable_switch_rate"],
        "reliable_switch_share": row["reliable_switch_share"]
        >= thresholds["minimum_reliable_switch_share"],
        "nonlocal_switch_count": row["nonlocal_switch_count"]
        <= thresholds["maximum_nonlocal_switch_count"],
        "reliable_distance_improvement_mean": row[
            "reliable_distance_improvement_mean"
        ] >= thresholds["minimum_distance_improvement"],
        "reliable_similarity_delta_mean": row[
            "reliable_similarity_delta_mean"
        ] >= thresholds["minimum_similarity_improvement"],
        "within_match_rate_delta": row["within_match_rate_delta"]
        >= thresholds["minimum_within_match_rate_delta"],
    }


def evaluate_assignment_gate(rows, thresholds=None):
    thresholds = dict(
        {
            "minimum_reliable_switch_count": 200,
            "minimum_reliable_switch_rate": 0.005,
            "maximum_reliable_switch_rate": 0.03,
            "minimum_reliable_switch_share": 0.95,
            "maximum_nonlocal_switch_count": 0,
            "minimum_distance_improvement": 3.0,
            "minimum_similarity_improvement": 0.1,
            "minimum_within_match_rate_delta": 0.0,
        },
        **(thresholds or {}),
    )
    if not rows:
        raise ValueError("at least one aggregate row is required")
    evaluated = []
    for row in rows:
        checks = _checks(row, thresholds)
        evaluated.append({"row": row, "checks": checks, "passed": all(checks.values())})
    passing = [item for item in evaluated if item["passed"]]
    candidates = passing or evaluated
    selected = max(
        candidates,
        key=lambda item: (
            sum(item["checks"].values()),
            item["row"]["reliable_switch_count"],
            item["row"]["reliable_switch_share"],
            item["row"]["reliable_distance_improvement_mean"],
            -item["row"]["lambda"],
        ),
    )
    failures = [name for name, passed in selected["checks"].items() if not passed]
    return {
        "gate_pass": bool(passing),
        "selected_lambda": float(selected["row"]["lambda"]),
        "failures": failures,
        "selected_metrics": selected["row"],
        "thresholds": thresholds,
        "per_lambda_checks": [
            {
                "lambda": float(item["row"]["lambda"]),
                "passed": item["passed"],
                "checks": item["checks"],
            }
            for item in evaluated
        ],
        "next_action": (
            "implement_detached_prototype_guided_hungarian_training"
            if passing
            else "do_not_modify_matcher_or_start_training"
        ),
    }


def _checkpoint_state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def _model_args(checkpoint, dataset, mean_std_path, num_workers):
    import train_p2p as p2p

    defaults = vars(p2p.get_args_parser().parse_args([]))
    saved = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved, Namespace):
        saved = vars(saved)
    if isinstance(saved, dict):
        defaults.update(saved)
    defaults.update(
        {
            "dataset": dataset,
            "mean_std_path": mean_std_path,
            "test_mean_std_path": mean_std_path,
            "eval_split": "test",
            "num_workers": int(num_workers),
            "batch_size": 1,
            "distributed": False,
            "world_size": 1,
            "rank": 0,
            "gpu": 0,
            "local_rank": None,
            "output_dir": "",
            "proto_inference_fusion": 0,
        }
    )
    return Namespace(**defaults)


def _slide_group(path):
    name = os.path.basename(str(path))
    return name.split(".kfb_", 1)[0] if ".kfb_" in name else name.split("_", 1)[0]


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args):
    import torch
    from torch.utils.data import DataLoader, Subset

    import train_p2p as p2p
    from dataset_zy_src import build_dataset
    from models.detr import build_model
    from prototype_audit_sampling import stratified_round_robin_indices
    from utils import collate_fn_pad

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = _model_args(
        checkpoint, args.dataset, args.mean_std_path, args.num_workers
    )
    p2p.args = model_args
    dataset = build_dataset(model_args, "test")
    selected_indices = stratified_round_robin_indices(
        dataset.files, int(args.max_images), _slide_group
    )
    selected_files = [dataset.files[index] for index in selected_indices]
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=model_args.num_workers,
        collate_fn=collate_fn_pad,
    )

    device = torch.device(f"cuda:{args.gpu}")
    model = build_model(model_args).to(device)
    result = model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    if model.prototype_mode != "prototype_reliability_rescue":
        raise RuntimeError(
            "checkpoint must use prototype_reliability_rescue, got "
            f"{model.prototype_mode!r}"
        )
    head = model.prototype_head
    if head is None or not bool(head.prototype_ready.item()):
        raise RuntimeError("checkpoint foreground prototypes are not ready")
    model.eval()

    lambdas = parse_lambdas(args.lambdas)
    per_image = []
    per_target_switch = []
    empty_images = 0
    with torch.no_grad():
        for image_index, (batch, file_path) in enumerate(zip(loader, selected_files)):
            images, gt_points_padded, labels, lengths = batch
            del labels
            gt_count = int(lengths[0])
            if gt_count <= 0:
                empty_images += 1
                continue
            images = images.to(device, non_blocking=True)
            gt_points = gt_points_padded[0].reshape(-1, 2)[:gt_count].to(
                device=device, dtype=torch.float32
            )

            anchors = model.get_aps(images)
            features = model.backbone(images)
            reg_features, cls_features, _, _ = model.extract_features(
                features, anchors
            )
            predicted_points = model.reg_head(reg_features) + anchors
            raw_logits = model.cls_head(cls_features)
            _, support_features, _, _ = model.extract_features(
                features, gt_points.unsqueeze(0)
            )

            candidate_embeddings = head._embed_detached(cls_features)[0]
            support_embeddings = head._embed_detached(support_features)[0]
            prototypes = head.normalized_prototypes()
            target_prototype_ids = support_embeddings.matmul(prototypes.t()).argmax(-1)
            target_prototypes = prototypes[target_prototype_ids]
            similarities = candidate_embeddings.matmul(target_prototypes.t())
            distances = torch.cdist(
                predicted_points[0],
                gt_points,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            scores = torch.softmax(raw_logits[0], dim=-1)[:, 0]
            class_cost = -(scores + 1e-8).log().unsqueeze(1).expand_as(distances)
            baseline_cost = (
                float(model_args.set_cost_point) * distances
                + float(model_args.set_cost_class) * class_cost
            )

            scores_np = scores.cpu().numpy()
            distances_np = distances.cpu().numpy()
            similarities_np = similarities.cpu().numpy()
            baseline_cost_np = baseline_cost.cpu().numpy()
            baseline_assignment = solve_target_assignment(baseline_cost_np)
            for prototype_weight in lambdas:
                guided_cost = build_guided_cost(
                    baseline_cost_np,
                    distances_np,
                    similarities_np,
                    prototype_weight,
                    args.local_radius,
                )
                guided_assignment = solve_target_assignment(guided_cost)
                switch_records = build_target_switch_records(
                    baseline_assignment,
                    guided_assignment,
                    scores_np,
                    distances_np,
                    similarities_np,
                    match_radius=args.match_radius,
                    minimum_distance_improvement=(
                        args.minimum_distance_improvement
                    ),
                    minimum_similarity_improvement=(
                        args.minimum_similarity_improvement
                    ),
                    minimum_candidate_score=args.minimum_candidate_score,
                    maximum_score_drop=args.maximum_score_drop,
                )
                per_target_switch.extend(
                    {
                        "lambda": float(prototype_weight),
                        "image": os.path.basename(str(file_path)),
                        **record,
                    }
                    for record in switch_records
                )
                metrics = compare_target_assignments(
                    baseline_assignment,
                    guided_assignment,
                    scores_np,
                    distances_np,
                    similarities_np,
                    score_threshold=args.score_threshold,
                    match_radius=args.match_radius,
                    distance_tolerance=args.distance_tolerance,
                    minimum_distance_improvement=(
                        args.minimum_distance_improvement
                    ),
                    minimum_similarity_improvement=(
                        args.minimum_similarity_improvement
                    ),
                    minimum_candidate_score=args.minimum_candidate_score,
                    maximum_score_drop=args.maximum_score_drop,
                )
                per_image.append(
                    {
                        "lambda": float(prototype_weight),
                        "image": os.path.basename(str(file_path)),
                        **metrics,
                    }
                )
            if (image_index + 1) % 25 == 0 or image_index + 1 == len(loader):
                print(
                    f"[Prototype-assignment-audit] processed={image_index + 1}/"
                    f"{len(loader)}",
                    flush=True,
                )

    aggregate_rows = aggregate_assignment_metrics(per_image)
    decision = evaluate_assignment_gate(aggregate_rows)
    decision.update(
        {
            "implementation_version": PROTOTYPE_GUIDED_ASSIGNMENT_AUDIT_VERSION,
            "diagnostic_only": True,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch")
            if isinstance(checkpoint, dict)
            else None,
            "dataset": os.path.abspath(args.dataset),
            "requested_images": int(args.max_images),
            "selected_images": len(selected_indices),
            "evaluated_non_empty_images": len(per_image) // len(lambdas),
            "empty_images": int(empty_images),
            "cost_formula": (
                "C=class_cost+0.1*point_distance+lambda*(1-local_prototype_similarity)"
            ),
            "prototype_term_detached": True,
            "reliable_switch_definition": {
                "maximum_guided_distance": float(args.match_radius),
                "minimum_distance_improvement": float(
                    args.minimum_distance_improvement
                ),
                "minimum_similarity_improvement": float(
                    args.minimum_similarity_improvement
                ),
                "minimum_guided_candidate_score": float(
                    args.minimum_candidate_score
                ),
                "maximum_score_drop": float(args.maximum_score_drop),
            },
            "warning": (
                "This is a source-test mechanism audit, not a publishable performance "
                "estimate or a deployable lambda selection."
            ),
        }
    )
    _write_csv(
        os.path.join(args.output_dir, "prototype_guided_assignment_per_image.csv"),
        per_image,
    )
    _write_csv(
        os.path.join(args.output_dir, "prototype_guided_assignment_switches.csv"),
        per_target_switch,
    )
    _write_csv(
        os.path.join(args.output_dir, "prototype_guided_assignment_sweep.csv"),
        aggregate_rows,
    )
    with open(
        os.path.join(args.output_dir, "prototype_guided_assignment_gate.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"[Prototype-assignment-audit] output={args.output_dir}", flush=True)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Audit detached prototype guidance for Hungarian assignment."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--lambdas", default="0.2,0.5,1.0,2.0")
    parser.add_argument("--local_radius", type=float, default=15.0)
    parser.add_argument("--match_radius", type=float, default=15.0)
    parser.add_argument("--distance_tolerance", type=float, default=1.0)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--minimum_distance_improvement", type=float, default=3.0)
    parser.add_argument("--minimum_similarity_improvement", type=float, default=0.1)
    parser.add_argument("--minimum_candidate_score", type=float, default=0.2)
    parser.add_argument("--maximum_score_drop", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main():
    args = get_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
