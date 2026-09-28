#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_topk"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_topk_onnx"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/topk"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --rows 3 --columns 17 --k 1
"${DIRECT_TEST}" --kernel-elf "${ELF}" --rows 3 --columns 17 --k 5
"${DIRECT_TEST}" --kernel-elf "${ELF}" --rows 3 --columns 17 --k 17

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported"
