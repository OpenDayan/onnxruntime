#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${DAYAN_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
DIRECT_TEST="${BUILD_DIR}/test_dayan_maxpool"
RUNNER="${BUILD_DIR}/test_dayan_onnx_model"
PLUGIN="${BUILD_DIR}/libdayan_ep.so"
DATA="${ROOT_DIR}/build/operators/maxpool"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,3,8,9 --kernel 2,2 --stride 2,2 --pad 0,0
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,3,8,9 --kernel 3,3 --stride 2,2 --pad 1,1
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,3,8,9 --kernel 5,5 --stride 1,1 --pad 2,2

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-dayan MaxPool
