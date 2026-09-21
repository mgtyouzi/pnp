#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/yh_1/p2p-set-pnp/p2p-src-zzh-2025}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
timestamp="$(date +%Y%m%d_%H%M%S)"
QUEUE_ROOT="${QUEUE_ROOT:-${PROJECT_ROOT}/experiments/p2p_proto_v22_gated_queue_${timestamp}_gpu${GPU}_bt${BATCH_SIZE}}"
RAPID_ROOT="${QUEUE_ROOT}/rapid"
FAITHFUL_ROOT="${QUEUE_ROOT}/faithful"

cd "${PROJECT_ROOT}"
mkdir -p "${QUEUE_ROOT}"

echo "[Queue 1/2] rapid mechanism gate"
env \
  MODE=rapid \
  GPU="${GPU}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PAIR_ROOT="${RAPID_ROOT}" \
  RUN_PREFLIGHT=1 \
  bash run_p2p_prototype_v22_paired_gpu3.sh

rapid_action="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["next_action"])' "${RAPID_ROOT}/prototype_v22_decision.json")"
if [ "${rapid_action}" != "run_faithful_paired_validation" ]; then
  echo "[STOP] rapid gate did not authorize faithful validation: ${rapid_action}"
  echo "[DECISION] ${RAPID_ROOT}/prototype_v22_decision.json"
  exit 3
fi

echo "[Queue 2/2] faithful full-epoch paired validation"
env \
  MODE=faithful \
  GPU="${GPU}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PAIR_ROOT="${FAITHFUL_ROOT}" \
  RUN_PREFLIGHT=0 \
  bash run_p2p_prototype_v22_paired_gpu3.sh

printf '%s\n' "${QUEUE_ROOT}" > "${PROJECT_ROOT}/experiments/latest_p2p_prototype_v22_gated_queue.txt"
echo "[SUCCESS] gated queue completed: ${QUEUE_ROOT}"
echo "[FINAL DECISION] ${FAITHFUL_ROOT}/prototype_v22_decision.json"
