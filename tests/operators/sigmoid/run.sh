#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_elementwise"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/sigmoid"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op sigmoid --shape 2,3,5
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op sigmoid --shape 1,2,3,16

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-rvnpu Sigmoid --atol 1e-4 --rtol 1e-4
