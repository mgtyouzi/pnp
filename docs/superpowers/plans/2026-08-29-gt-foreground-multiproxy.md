# GT-Guided Foreground Multi-Proxy P2P Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate an isolated training-only foreground multi-proxy regularizer for P2P.

**Architecture:** A new prototype mode consumes original candidate classification features plus GT-point support features from the same P2P multi-scale extractor. It maintains a momentum projector and epoch-level non-parametric foreground prototypes, while leaving detector inference unchanged.

**Tech Stack:** Python 3.8, PyTorch 1.10, single-GPU CUDA training, existing P2P matcher and no-empty protocol.

## Global Constraints

- Do not modify legacy prototype behavior.
- Do not add HBS or inference fusion.
- Write a failing test before each production behavior.
- Run paired control and prototype training from the same checkpoint and RNG.

---

### Task 1: Tensor-level foreground proxy module

**Files:**
- Create: `models/gt_foreground_proxy.py`
- Create: `test_gt_foreground_proxy.py`

**Interfaces:**
- `GTForegroundProxy.forward(query_features, raw_logits, support_features)` returns query and support embeddings without changing logits.
- `compute_loss_and_cache(...)` returns a scalar loss and numeric diagnostics.
- `begin_epoch()`, `maybe_refresh_prototypes()`, and `finalize_epoch()` match existing training hooks.

- [x] Write failing tests for reliable query gating, near/far background selection, warm-up loss, initialization, EMA update, dead-proxy recovery, and inference identity.
- [x] Run the no-Torch structure test and verify failure because the module is absent.
- [x] Implement the module behavior.
- [ ] Run `python test_gt_foreground_proxy.py` in the remote Torch environment.

### Task 2: P2P feature-path integration

**Files:**
- Modify: `models/detr.py`
- Modify: `train_p2p.py`
- Create: `test_gt_foreground_proxy_wiring.py`

**Interfaces:**
- `DETR.forward(images, prototype_support_points=None)` pads valid GT points and samples support classification features only for the new mode.
- The training loop passes GT points only when the new mode is active and dispatches its loss through the existing prototype hooks.

- [x] Write failing wiring tests for mode construction, support padding, unchanged inference logits, and train-loop dispatch markers.
- [x] Run `python test_gt_foreground_proxy_wiring.py` and verify expected failures.
- [x] Add the new mode, arguments, support extraction, outputs, and loss dispatch.
- [ ] Run the dynamic new and legacy prototype tests remotely.

### Task 3: GPU 4 paired mechanism experiment

**Files:**
- Create: `run_gt_foreground_proxy_paired_gpu4.sh`

**Interfaces:**
- Runs preflight, a 10-epoch paired control, then a 10-epoch prototype run on physical GPU 4.
- Saves checkpoints, console logs, mechanism diagnostics, and the exact run configuration under one pair root.

- [x] Add path and version preflight checks.
- [x] Lock common P2P settings and initialization across paired runs.
- [x] Configure the approved foreground-proxy defaults and disable inference fusion.
- [ ] Validate with `bash -n run_gt_foreground_proxy_paired_gpu4.sh`.

### Task 4: Verification

- [ ] Run `python -m py_compile models/gt_foreground_proxy.py models/detr.py train_p2p.py`.
- [ ] Run all new tests plus `python test_p2p_prototype.py`.
- [ ] Check the shell script syntax.
- [ ] Inspect the diff and verify no existing prototype mode changed behavior.
