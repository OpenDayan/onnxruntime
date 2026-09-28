#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DATA="${ROOT_DIR}/build/operators/sub"

echo "[1/2] Direct kernel 数值测试"
"${ROOT_DIR}/build/linux/test_rvnpu_elementwise" \
  --kernel-elf "${ELF}" --op sub --shape 1,2,16 --rhs-shape 1,2,16

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${ROOT_DIR}/build/linux/test_rvnpu_onnx_model" \
  --model "${DATA}/supported/model.onnx" \
  --plugin "${ROOT_DIR}/build/linux/librvnpu_ep.so" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-rvnpu Sub
