#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:?RVNPU_KERNEL_ELF is required}"
PROBE="${ROOT_DIR}/build/linux/rvnpu_matmul_probe"
RUNNER="${ROOT_DIR}/build/linux/rvnpu_ort_runner"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
MODEL_DIR="${ROOT_DIR}/build/models"

echo "[1/2] Direct kernel 数值测试"
"${PROBE}" --kernel-elf "${ELF}" --m 1 --n 32 --k 32
"${PROBE}" --kernel-elf "${ELF}" --m 32 --n 32 --k 32
"${PROBE}" --kernel-elf "${ELF}" --m 3 --n 5 --k 7

echo
echo "[2/2] tiny CNN：CPU/reference/RVNPU 输出对比"
"${RUNNER}" --model "${MODEL_DIR}/tiny_cnn.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${MODEL_DIR}" --repeat 2 --compare-cpu
