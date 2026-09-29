# Prototype Metric E1 Snapshot

This directory records the E1 experiment that trained on the paraffin train split and evaluated on the paraffin test split.

## Contents

- `source_overlay/` contains the E1 source files captured from the experiment tree, retaining their original relative paths. Machine-specific paths in the smoke helper and baseline compatibility test are parameterized for sharing.
- `run_config.json` records the training and evaluation settings without machine-specific absolute paths.
- `test_summary.json` records the aggregate evaluation result. Per-image CSVs, image data, logs, and model weights are intentionally excluded.

## Important provenance note

The experiment was run from a separate copy of the older `p2p-yfh/p2p-src-zzh-2025` baseline. Its `train_p2p.py` and `models/detr.py` differ substantially from the current `pnp` main tree. This is an archival source overlay, not a patch to apply over the current `pnp` files. Keep it isolated unless the exact base revision is recovered and the changes are ported and tested against that revision.

## Result

The final checkpoint was epoch index 99. Strict checkpoint loading reported zero missing and zero unexpected keys. On 876 non-empty test images, E1 achieved precision 0.6156, recall 0.7535, F1 0.6776, and MAE 6.5090. One empty-GT image was skipped according to the recorded evaluation protocol.

The previously reported paraffin baseline F1 around 0.760 should only be compared after confirming identical evaluator and threshold settings. The model checkpoint and patient/image-level outputs are not included in this snapshot.
