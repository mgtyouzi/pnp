# Frozen Prototype Candidate Separation Design

## Goal

Determine whether the frozen supervised prototype bank separates true cell
detections from difficult false-positive detections inside the original P2P
cell-positive candidate set. This is a diagnostic only: it must not train or
modify P2P, HBS, the projector, or the prototype bank.

## Evaluation contract

- Reuse the 0318 detector checkpoint and its SHA-matched frozen bank.
- Reuse the existing P2P border filtering, class reservation, deduplication,
  greedy point matching, `match_dis=15`, and `near_radius=30` protocol.
- Label retained raw-cell candidates as `tp`, `duplicate_fp`, `near_miss_fp`,
  `background_far_fp`, or `empty_image_fp`.
- Represent each low-score classification FN by its nearest raw-background
  candidate within `near_radius`, labelled `low_score_fn`.
- Report raw classification margin, prototype margin, and a source-fitted joint
  score for every labelled candidate.
- Select all joint-score normalization, mixture, and decision thresholds on
  paraffin source labels only. Frozen labels may evaluate fixed parameters but
  must never tune them.

## Source calibration

For `tp` versus `background_far_fp`, standardize raw and prototype margins using
source statistics. Search a fixed beta grid for
`joint=(1-beta)*raw_z + beta*prototype_z`, selecting the largest source AUC with
the smaller beta as tie-breaker. Select the balanced-accuracy threshold on the
same source candidates. Save detector/bank hashes and all calibration values.

## Outputs

- Candidate-level CSV with image, slide, category, raw margin, prototype margin,
  and joint score.
- JSON/CSV separation report for TP versus background-far FP, near-miss FP, and
  all FP using global and slide-level AUC.
- Source calibration JSON.
- Target decision JSON using fixed source calibration:
  - `keep_and_build_gate` when target prototype AUC is at least 0.65;
  - `weak_diagnostic_only` for AUC in [0.55, 0.65);
  - `stop_current_prototype_bank` below 0.55.

## Failure handling

Fail before inference on checkpoint/bank/calibration identity mismatch. Fail when
source calibration lacks TP or background-far FP candidates. Empty comparison
groups produce explicit `NaN` metrics rather than silently changing labels.
