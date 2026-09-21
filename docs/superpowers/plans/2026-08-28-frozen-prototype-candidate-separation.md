# Frozen Prototype Candidate Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a source-calibrated candidate-level audit that decides whether the frozen prototype bank distinguishes true P2P detections from difficult background false positives.

**Architecture:** Pure NumPy helpers reproduce the existing P2P matching protocol and fit a source-only joint score. A separate GPU runner reuses the existing frozen-bank model loader and record extractor for source calibration and fixed target evaluation.

**Tech Stack:** Python 3.8, NumPy, PyTorch 1.10, existing P2P diagnostic utilities.

## Global Constraints

- No detector, HBS, projector, or prototype training.
- No target-label parameter selection.
- Preserve existing P2P deduplication and matching semantics.
- Validate checkpoint and frozen-bank hashes before inference.

---

### Task 1: Pure candidate labelling

**Files:**
- Modify: `frozen_proto_rescoring.py`
- Modify: `test_frozen_proto_rescoring.py`

**Interfaces:**
- Produces: `label_candidate_diagnostics(points, raw_logits, prototype_logits, gt_points, dedup_interval, match_dis, near_radius) -> list[dict]`.

- [ ] Add a failing toy test containing TP, duplicate FP, near-miss FP, background-far FP, and low-score FN candidates.
- [ ] Verify the test fails because the helper is absent.
- [ ] Implement index-preserving deduplication and exact candidate labels.
- [ ] Run `python test_frozen_proto_rescoring.py` and require PASS.

### Task 2: Source-only joint calibration

**Files:**
- Modify: `frozen_proto_rescoring.py`
- Modify: `test_frozen_proto_rescoring.py`

**Interfaces:**
- Produces: `fit_candidate_joint_calibration(rows, beta_grid) -> dict`.
- Produces: `apply_candidate_joint_calibration(rows, calibration) -> ndarray`.
- Produces: `candidate_separation_report(rows, score_names) -> list[dict]`.

- [ ] Add failing tests proving beta and normalization come only from source rows.
- [ ] Add a failing calibration identity test.
- [ ] Implement rank-aware AUC, balanced-accuracy threshold selection, fixed target application, and grouped reports.
- [ ] Run the complete pure test file and require PASS.

### Task 3: GPU diagnostic runner

**Files:**
- Create: `diagnose_frozen_proto_candidate_separation.py`
- Modify: `test_frozen_proto_rescoring.py`

**Interfaces:**
- CLI modes: `source_calibrate` and `target_fixed`.
- Reuses: `_load_model` and `_extract_records` from `diagnose_frozen_proto_suppression.py`.

- [ ] Add a failing AST test that target calibration loads before record extraction and target mode never calls the source fitter.
- [ ] Implement source candidate export, source calibration, fixed target audit, and decision output.
- [ ] Write explicit CSV headers for empty groups.
- [ ] Run pure tests and Python 3.8-compatible syntax compilation.

### Task 4: Final verification

**Files:**
- Verify all files above.

- [ ] Run `python test_frozen_proto_rescoring.py`.
- [ ] Run `python test_frozen_bank_transfer_audit.py`.
- [ ] Run `python -m py_compile frozen_proto_rescoring.py diagnose_frozen_proto_suppression.py diagnose_frozen_proto_candidate_separation.py test_frozen_proto_rescoring.py`.
- [ ] Provide exact upload, source-calibration, and fixed-target GPU 3 commands.
