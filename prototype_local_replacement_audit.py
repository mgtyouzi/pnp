import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from prototype_guided_assignment_audit import (
    _checkpoint_state_dict,
    _model_args,
    _slide_group,
    safe_div,
    solve_target_assignment,
)


PROTOTYPE_LOCAL_REPLACEMENT_AUDIT_VERSION = (
    "prototype_local_replacement_audit_v1_20260907"
)


def analyze_local_replacement_candidates(
    baseline_assignment,
    candidate_scores,
    point_distances,
    prototype_similarities,
    match_radius=15.0,
    minimum_distance_improvement=3.0,
    minimum_similarity_improvement=0.1,
    minimum_candidate_score=0.2,
    maximum_score_drop=0.25,
    distance_utility_weight=1.0,
    similarity_utility_weight=1.0,
    score_utility_weight=0.5,
):
    baseline = np.asarray(baseline_assignment, dtype=np.int64).reshape(-1)
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    distances = np.asarray(point_distances, dtype=np.float64)
    similarities = np.asarray(prototype_similarities, dtype=np.float64)
    if distances.shape != similarities.shape:
        raise ValueError("distance and prototype-similarity matrices must match")
    if distances.shape[0] != len(scores) or distances.shape[1] != len(baseline):
        raise ValueError("candidate and target dimensions are inconsistent")
    if min(match_radius, minimum_distance_improvement) <= 0:
        raise ValueError("match radius and distance improvement must be positive")
    if minimum_similarity_improvement < 0 or maximum_score_drop < 0:
        raise ValueError("similarity improvement and score drop must be non-negative")

    assigned = np.zeros(len(scores), dtype=bool)
    assigned[baseline] = True
    unassigned_indices = np.flatnonzero(~assigned)
    funnel = {
        "unassigned_candidates": int(len(unassigned_indices)),
        "nearest_target_local": 0,
        "distance_improvement_pass": 0,
        "similarity_improvement_pass": 0,
        "candidate_score_floor_pass": 0,
        "score_drop_pass": 0,
        "eligible_proposals": 0,
    }
    proposals = []
    if not len(baseline) or not len(unassigned_indices):
        return proposals, funnel

    # Every candidate may propose only to its nearest GT. This prevents a local
    # correction from stealing a candidate across neighboring cells.
    nearest_targets = distances[unassigned_indices].argmin(axis=1)
    for candidate_index, target_index in zip(unassigned_indices, nearest_targets):
        candidate_index = int(candidate_index)
        target_index = int(target_index)
        candidate_distance = float(distances[candidate_index, target_index])
        if candidate_distance > float(match_radius):
            continue
        funnel["nearest_target_local"] += 1

        baseline_candidate = int(baseline[target_index])
        baseline_distance = float(distances[baseline_candidate, target_index])
        distance_improvement = baseline_distance - candidate_distance
        if distance_improvement < float(minimum_distance_improvement):
            continue
        funnel["distance_improvement_pass"] += 1

        baseline_similarity = float(
            similarities[baseline_candidate, target_index]
        )
        candidate_similarity = float(
            similarities[candidate_index, target_index]
        )
        similarity_delta = candidate_similarity - baseline_similarity
        if similarity_delta < float(minimum_similarity_improvement):
            continue
        funnel["similarity_improvement_pass"] += 1

        baseline_score = float(scores[baseline_candidate])
        candidate_score = float(scores[candidate_index])
        if candidate_score < float(minimum_candidate_score):
            continue
        funnel["candidate_score_floor_pass"] += 1
        score_delta = candidate_score - baseline_score
        if score_delta < -float(maximum_score_drop):
            continue
        funnel["score_drop_pass"] += 1

        utility = (
            float(distance_utility_weight)
            * distance_improvement
            / float(match_radius)
            + float(similarity_utility_weight) * similarity_delta
            + float(score_utility_weight) * score_delta
        )
        proposals.append(
            {
                "target_index": target_index,
                "baseline_candidate": baseline_candidate,
                "candidate_index": candidate_index,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "score_delta": score_delta,
                "baseline_distance": baseline_distance,
                "candidate_distance": candidate_distance,
                "distance_improvement": distance_improvement,
                "baseline_similarity": baseline_similarity,
                "candidate_similarity": candidate_similarity,
                "similarity_delta": similarity_delta,
                "utility": float(utility),
            }
        )
    funnel["eligible_proposals"] = len(proposals)
    return proposals, funnel


def build_local_replacement_proposals(*args, **kwargs):
    proposals, _ = analyze_local_replacement_candidates(*args, **kwargs)
    return proposals


def select_conflict_free_replacements(proposals):
    selected = []
    used_targets = set()
    used_candidates = set()
    ordered = sorted(
        proposals,
        key=lambda item: (
            -float(item["utility"]),
            int(item["target_index"]),
            int(item["candidate_index"]),
        ),
    )
    for proposal in ordered:
        target_index = int(proposal["target_index"])
        candidate_index = int(proposal["candidate_index"])
        if target_index in used_targets or candidate_index in used_candidates:
            continue
        selected.append(proposal)
        used_targets.add(target_index)
        used_candidates.add(candidate_index)
    return sorted(selected, key=lambda item: int(item["target_index"]))


def _mean(records, key):
    return safe_div(sum(float(record[key]) for record in records), len(records))


def summarize_local_replacements(selected, targets, unassigned_candidates):
    count = len(selected)
    score_improved = sum(float(item["score_delta"]) > 0 for item in selected)
    low_to_high = sum(
        float(item["baseline_score"]) < 0.5 <= float(item["candidate_score"])
        for item in selected
    )
    return {
        "targets": int(targets),
        "unassigned_candidates": int(unassigned_candidates),
        "replacement_count": count,
        "replacement_rate": safe_div(count, targets),
        "score_improved_count": int(score_improved),
        "score_improved_rate": safe_div(score_improved, count),
        "low_to_high_count": int(low_to_high),
        "mean_score_delta": _mean(selected, "score_delta"),
        "mean_distance_improvement": _mean(selected, "distance_improvement"),
        "mean_similarity_delta": _mean(selected, "similarity_delta"),
        "mean_baseline_distance": _mean(selected, "baseline_distance"),
        "mean_candidate_distance": _mean(selected, "candidate_distance"),
        "mean_baseline_score": _mean(selected, "baseline_score"),
        "mean_candidate_score": _mean(selected, "candidate_score"),
    }


def evaluate_local_replacement_gate(summary, thresholds=None):
    thresholds = dict(
        {
            "minimum_replacement_count": 200,
            "minimum_replacement_rate": 0.005,
            "maximum_replacement_rate": 0.03,
            "minimum_mean_distance_improvement": 3.0,
            "minimum_mean_similarity_delta": 0.1,
            "minimum_mean_score_delta": -0.05,
        },
        **(thresholds or {}),
    )
    checks = {
        "replacement_count": summary["replacement_count"]
        >= thresholds["minimum_replacement_count"],
        "minimum_replacement_rate": summary["replacement_rate"]
        >= thresholds["minimum_replacement_rate"],
        "maximum_replacement_rate": summary["replacement_rate"]
        <= thresholds["maximum_replacement_rate"],
        "mean_distance_improvement": summary["mean_distance_improvement"]
        >= thresholds["minimum_mean_distance_improvement"],
        "mean_similarity_delta": summary["mean_similarity_delta"]
        >= thresholds["minimum_mean_similarity_delta"],
        "mean_score_delta": summary["mean_score_delta"]
        >= thresholds["minimum_mean_score_delta"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "gate_pass": not failures,
        "failures": failures,
        "checks": checks,
        "metrics": summary,
        "thresholds": thresholds,
        "next_action": (
            "implement_post_hungarian_local_replacement_matcher"
            if not failures
            else "do_not_start_detector_training"
        ),
    }


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    all_proposals = []
    all_selected = []
    per_image = []
    funnel_totals = defaultdict(int)
    total_targets = 0
    empty_images = 0
    with torch.no_grad():
        for image_index, (batch, file_path) in enumerate(zip(loader, selected_files)):
            images, gt_points_padded, labels, lengths = batch
            del labels
            gt_count = int(lengths[0])
            if gt_count <= 0:
                empty_images += 1
                continue
            total_targets += gt_count
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
            baseline_assignment = solve_target_assignment(
                baseline_cost.cpu().numpy()
            )
            proposals, funnel = analyze_local_replacement_candidates(
                baseline_assignment,
                scores_np,
                distances_np,
                similarities_np,
                match_radius=args.match_radius,
                minimum_distance_improvement=args.minimum_distance_improvement,
                minimum_similarity_improvement=(
                    args.minimum_similarity_improvement
                ),
                minimum_candidate_score=args.minimum_candidate_score,
                maximum_score_drop=args.maximum_score_drop,
                distance_utility_weight=args.distance_utility_weight,
                similarity_utility_weight=args.similarity_utility_weight,
                score_utility_weight=args.score_utility_weight,
            )
            selected = select_conflict_free_replacements(proposals)
            image_name = os.path.basename(str(file_path))
            all_proposals.extend({"image": image_name, **item} for item in proposals)
            all_selected.extend({"image": image_name, **item} for item in selected)
            for key, value in funnel.items():
                funnel_totals[key] += int(value)
            image_summary = summarize_local_replacements(
                selected,
                targets=gt_count,
                unassigned_candidates=funnel["unassigned_candidates"],
            )
            per_image.append({"image": image_name, **image_summary})
            if (image_index + 1) % 25 == 0 or image_index + 1 == len(loader):
                print(
                    f"[Prototype-local-replacement] processed={image_index + 1}/"
                    f"{len(loader)}, proposals={len(all_proposals)}, "
                    f"selected={len(all_selected)}",
                    flush=True,
                )

    summary = summarize_local_replacements(
        all_selected,
        targets=total_targets,
        unassigned_candidates=funnel_totals["unassigned_candidates"],
    )
    summary.update(
        {
            "implementation_version": PROTOTYPE_LOCAL_REPLACEMENT_AUDIT_VERSION,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch")
            if isinstance(checkpoint, dict)
            else None,
            "dataset": os.path.abspath(args.dataset),
            "selected_images": len(selected_indices),
            "evaluated_non_empty_images": len(per_image),
            "empty_images": empty_images,
            "proposal_count": len(all_proposals),
            "proposal_to_replacement_rate": safe_div(
                len(all_selected), len(all_proposals)
            ),
            "selection_funnel": dict(funnel_totals),
            "diagnostic_only": True,
            "warning": (
                "Source-test mechanism audit only; no matcher or checkpoint was modified."
            ),
        }
    )
    decision = evaluate_local_replacement_gate(summary)
    _write_csv(
        os.path.join(args.output_dir, "local_replacement_proposals.csv"),
        all_proposals,
    )
    _write_csv(
        os.path.join(args.output_dir, "local_replacement_selected.csv"),
        all_selected,
    )
    _write_csv(
        os.path.join(args.output_dir, "local_replacement_per_image.csv"),
        per_image,
    )
    with open(
        os.path.join(args.output_dir, "local_replacement_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with open(
        os.path.join(args.output_dir, "local_replacement_decision.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"[Prototype-local-replacement] output={args.output_dir}", flush=True)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Audit conflict-free post-Hungarian prototype replacements."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mean_std_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--match_radius", type=float, default=15.0)
    parser.add_argument("--minimum_distance_improvement", type=float, default=3.0)
    parser.add_argument("--minimum_similarity_improvement", type=float, default=0.1)
    parser.add_argument("--minimum_candidate_score", type=float, default=0.2)
    parser.add_argument("--maximum_score_drop", type=float, default=0.25)
    parser.add_argument("--distance_utility_weight", type=float, default=1.0)
    parser.add_argument("--similarity_utility_weight", type=float, default=1.0)
    parser.add_argument("--score_utility_weight", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main():
    run_audit(get_parser().parse_args())


if __name__ == "__main__":
    main()
