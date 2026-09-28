#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_elementwise"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/mul"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op mul --shape 2,3,5 --rhs-shape 2,3,5
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op mul --shape 1,3,4,5 \
  --rhs-shape 1,3,1,1 --swap-inputs

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" --expect-rvnpu Mul
