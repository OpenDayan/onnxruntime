#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DATA="${ROOT_DIR}/build/operators/resize/supported"
echo "[1/2] Direct kernel 数值测试"
"${ROOT_DIR}/build/linux/test_rvnpu_transform" --kernel-elf "${ELF}" --op resize
echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${ROOT_DIR}/build/linux/test_rvnpu_onnx_model" --model "${DATA}/model.onnx" \
  --plugin "${ROOT_DIR}/build/linux/librvnpu_ep.so" --kernel-elf "${ELF}" \
  --data-dir "${DATA}" --expect-rvnpu Resize
