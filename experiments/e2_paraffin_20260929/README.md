# E2 Paraffin Experiments (2026-09-29)

This folder archives both requested experiments, E2-A and E2-B, plus the exact E2 runtime source snapshot used by the isolated server project.

## Results

Metrics below come from the same deterministic slide-group holdout carved from the paraffin **training set**. They are not official held-out test results. See [validation_protocol.json](validation_protocol.json) for the leakage caveat.

| Run/checkpoint | Mode | Residual | P | R | F1 | MAE | Delta F1 vs initial split baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Initial P2P checkpoint | Baseline | off | 0.710625 | 0.770146 | 0.739189 | 5.715622 | 0 |
| E2-A best, epoch 11 | Frozen P2P | on | 0.712451 | 0.768161 | 0.739258 | 5.715775 | +0.000069 |
| E2-A final, epoch 24 | Frozen P2P | on | 0.713987 | 0.765148 | 0.738683 | 5.714379 | -0.000507 |
| E2-B best overall, epoch 10 | Online P2P | off | 0.651151 | 0.815139 | 0.723975 | 5.728604 | -0.015215 |
| E2-B best residual-active, epoch 39 | Online P2P | on | 0.640386 | 0.828417 | 0.722365 | 5.701411 | -0.016824 |

**Interpretation:** E2-A preserved the fixed detector and produced only a negligible peak change (+0.000069 F1); its final epoch was slightly below the initial score. E2-B initialized from the same baseline weights but started at epoch 0 with a fresh optimizer and updated the detector. Its best score occurred before residual activation, and its best residual-active score remained below the initial baseline. Neither run has been evaluated on the official test set in this bundle.

## Experiment Definitions

- **E2-A:** baseline detector frozen; E2 adapter trained for 25 epochs, learning rate 1e-4.
- **E2-B:** detector trained online from the baseline initialization for 40 epochs, learning rate 1e-5; this is fine-tuning from baseline weights, not random initialization or optimizer-state resume.
- Both used seed 0, batch size 1, weight decay 1e-4, and the same E2 settings in their run configurations.

Per-epoch metrics, E2 diagnostics, and training logs are under [A/](A/) and [B/](B/). Weights, dataset files, absolute server paths, and the full split manifest containing image filenames were excluded.

## Source Snapshot

Files under [source_snapshot/](source_snapshot/) are the exact runtime source files from the isolated E2 server project. They are archived separately and **do not overwrite repository-root code**. Four core files already on `main` differ from the server runtime versions; see [source_manifest.json](source_manifest.json) before applying or comparing them. The E2 adapter unit test and Python compile check passed in the server environment before upload.
