# Frozen Prototype One-Way Suppression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-calibrated, inference-only prototype background suppression without allowing background candidates to become cells.

**Architecture:** Pure NumPy functions own fusion, threshold generation, parameter selection, and decisions. The existing GPU diagnostic performs one detector forward per image, caches compact prediction records, and runs either source calibration or fixed target evaluation on CPU.

**Tech Stack:** Python 3.8, NumPy, PyTorch 1.10, existing P2P and frozen supervised prototype bank.

## Global Constraints

- No detector, HBS, or prototype training changes.
- No target-label parameter tuning.
- Alpha zero must remain an exact control.
- Raw background logits must remain unchanged.
- Detector checkpoint and bank hashes must be validated.

---

### Task 1: Pure one-way suppression and calibration

**Files:**
- Modify: `frozen_proto_rescoring.py`
- Modify: `test_frozen_proto_rescoring.py`

- [ ] Add failing tests for identity, no background-to-cell flips, selective cell suppression, threshold quantiles, and constrained source selection.
- [ ] Run `python test_frozen_proto_rescoring.py` and verify the new tests fail for missing interfaces.
- [ ] Implement `suppress_raw_cell_logits`, `thresholds_from_raw_cell_quantiles`, and `select_source_suppression`.
- [ ] Run the tests and verify they pass.

### Task 2: Source and target diagnostic modes

**Files:**
- Modify: `diagnose_frozen_proto_rescoring.py`
- Modify: `test_frozen_proto_rescoring.py`

- [ ] Add failing source-contract tests for `source_calibrate` and `target_fixed` modes.
- [ ] Refactor the diagnostic to cache one forward pass and evaluate the one-way grid on CPU.
- [ ] Write a source calibration JSON with hashes, selected parameters, constraints, and metrics.
- [ ] In target mode load only the fixed calibration pair, validate identities, and emit control-versus-fixed metrics.
- [ ] Verify empty samples, alpha-zero identity, and zero background-to-cell flips.

### Task 3: Verification

**Files:**
- Test: `test_frozen_proto_rescoring.py`
- Test: `test_frozen_bank_transfer_audit.py`

- [ ] Run both test scripts.
- [ ] Run `python -m py_compile` on changed Python files.
- [ ] Parse changed files with Python 3.8 grammar.
- [ ] Inspect command help and ensure target mode requires a calibration file.

