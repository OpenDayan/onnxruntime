#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:?RVNPU_KERNEL_ELF is required}"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"

"${RUNNER}" \
  --model "${MODEL_DIR}/yolov8n.onnx" \
  --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" \
  --data-dir "${MODEL_DIR}/data" \
  --expect-rvnpu-counts Conv=64,Add=6,Mul=57,Sigmoid=58,MaxPool=3 \
  --report-max-error-percent \
  --atol 1e-3 --rtol 1e-3
