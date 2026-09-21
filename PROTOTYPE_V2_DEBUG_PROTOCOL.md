# Prototype-v2 mechanism debug protocol

This protocol isolates the prototype branch before another long training run.

## Outputs

- `checkpoint_initialization_report.json`: every loaded, missing, skipped, and shape-mismatched checkpoint key.
- `run_initialization.json`: RNG fingerprint and exact SHA-256 hash for each parameter group.
- `first_batch_debug.jsonl`: exact input, raw P2P output, base-loss, and matcher fingerprints before each epoch update.
- `dataset_keypoint_safety.json`: real augmentation failures, replacements, clipped/dropped source points.
- `run_debug.jsonl`: per-epoch RNG, sample trace, P2P losses, gradient clipping, and parameter statistics.
- `prototype_step_debug.jsonl`: candidate counts, score distributions, margins, assignments, and gradient relation.
- `prototype_debug.jsonl`: queue state, center occupancy, drift, reinitialization, and full prototype similarities.
- `prototype_epoch_metrics.csv`: compact epoch-level mechanism table.
- `comparison/paired_debug_report.md`: paired control/prototype attribution verdict.

## Required gates

1. Data gate: no transform errors, replacement samples, or crop-size errors. One known invalid source point may be deterministically dropped, but its full record must remain in the audit and both paired runs must report identical sanitization counts.
2. Attribution gate: identical base parameter SHA-256, sample traces, and RNG states.
3. Sampling gate: accepted positives plus both hard and random backgrounds are present.
4. Gradient gate: prototype/classification gradient ratio is measurable; strong negative cosine is flagged.
5. Prototype gate: occupancy, effective center count, drift, and the exact worst foreground/background pair are recorded.

Do not start a 100/200-epoch run unless the data and attribution gates pass. A high cross-class cosine is only treated as a real collision when both centers also have meaningful assignment share.
