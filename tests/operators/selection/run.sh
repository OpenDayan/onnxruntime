#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${DAYAN_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
RUNNER="${BUILD_DIR}/test_dayan_onnx_model"
PLUGIN="${BUILD_DIR}/libdayan_ep.so"
DATA="${ROOT_DIR}/build/operators/selection"

run_case() {
  local directory="$1"
  local operator="$2"
  echo "[${operator}] ONNX Runtime 分配与数值测试"
  "${RUNNER}" --model "${DATA}/${directory}/model.onnx" \
    --plugin "${PLUGIN}" --kernel-elf "${ELF}" \
    --data-dir "${DATA}/${directory}" --expect-dayan "${operator}"
}

run_case argmax ArgMax
run_case gather_nd_int64 GatherND
run_case transpose_int64 Transpose
run_case transpose_int64_vector Transpose
run_case slice_int64 Slice
run_case slice_int64_reverse Slice

echo "[empty postprocess] 动态空张量链路测试"
"${RUNNER}" --model "${DATA}/empty_postprocess/model.onnx" \
  --plugin "${PLUGIN}" --kernel-elf "${ELF}" \
  --data-dir "${DATA}/empty_postprocess" \
  --expect-dayan-counts \
  Transpose=1,GatherND=1,Div=1,Concat=1,Slice=1,NonMaxSuppression=1
