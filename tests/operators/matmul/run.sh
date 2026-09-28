#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_matmul"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_tiny_cnn"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
MODEL_DIR="${ROOT_DIR}/build/models"

echo "[1/3] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --m 1 --n 32 --k 32
"${DIRECT_TEST}" --kernel-elf "${ELF}" --m 32 --n 32 --k 32
"${DIRECT_TEST}" --kernel-elf "${ELF}" --m 3 --n 5 --k 7

echo
echo "[2/3] MatMul 端到端性能测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --m 64 --n 64 --k 64 \
  --benchmark --warmup 1 --iterations 5

echo
echo "[3/3] tiny CNN：CPU/RVNPU 输出对比"
"${RUNNER}" --model "${MODEL_DIR}/tiny_cnn.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${MODEL_DIR}"
