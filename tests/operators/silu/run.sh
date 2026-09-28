#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_elementwise"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/silu"

echo "[1/2] Direct SiLU kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op silu --shape 1,2,3,16

echo
echo "[2/2] Sigmoid+Mul 自动融合与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --enabled-ops Silu --expect-rvnpu-counts Sigmoid=1,Mul=1 \
  --atol 1e-4 --rtol 1e-4
"${RUNNER}" --model "${DATA}/supported_swapped/model.onnx" \
  --plugin "${PLUGIN}" --kernel-elf "${ELF}" \
  --data-dir "${DATA}/supported_swapped" --enabled-ops Silu \
  --expect-rvnpu-counts Sigmoid=1,Mul=1 --atol 1e-4 --rtol 1e-4
