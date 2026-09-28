#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_conv_maxpool"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/maxpool"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op maxpool \
  --input-shape 1,3,8,9 --kernel 2,2 --stride 2,2 --pad 0,0
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op maxpool \
  --input-shape 1,3,8,9 --kernel 3,3 --stride 2,2 --pad 1,1
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op maxpool \
  --input-shape 1,3,8,9 --kernel 5,5 --stride 1,1 --pad 2,2

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-rvnpu MaxPool
