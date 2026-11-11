#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${RVNPU_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
DATA="${ROOT_DIR}/build/operators/sub"

echo "[1/2] Direct kernel 数值测试"
"${BUILD_DIR}/test_rvnpu_elementwise" \
  --kernel-elf "${ELF}" --op sub --shape 1,2,16 --rhs-shape 1,2,16

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${BUILD_DIR}/test_rvnpu_onnx_model" \
  --model "${DATA}/supported/model.onnx" \
  --plugin "${BUILD_DIR}/librvnpu_ep.so" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-rvnpu Sub
