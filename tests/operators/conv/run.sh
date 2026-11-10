#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_conv"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/conv"
REFERENCE="${RVNPU_CONV_REFERENCE:-fp32}"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,3,64,64 --weight-shape 16,3,3,3 \
  --stride 2,2 --pad 1,1 --bias --reference "${REFERENCE}"
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,16,32,32 --weight-shape 32,16,3,3 \
  --stride 1,1 --pad 1,1 --reference "${REFERENCE}"
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,64,20,20 --weight-shape 64,64,1,1 \
  --stride 1,1 --pad 0,0 --bias --reference "${REFERENCE}"
"${DIRECT_TEST}" --kernel-elf "${ELF}" \
  --input-shape 1,3,15,17 --weight-shape 5,3,3,3 \
  --stride 2,2 --pad 1,1 --bias --reference "${REFERENCE}"

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-rvnpu Conv --atol 2e-2 --rtol 2e-2
