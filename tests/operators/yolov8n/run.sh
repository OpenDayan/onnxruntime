#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:?RVNPU_KERNEL_ELF is required}"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_onnx_model"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"

ALL_COUNTS="Conv=64,Add=8,Mul=58,Sub=2,Div=1,Sigmoid=58,MaxPool=3,Concat=17,Split=8,Slice=2,Resize=2,Transpose=1,Softmax=1"
ENABLED_OPERATOR="${RVNPU_YOLO_OPERATOR:-}"
if [[ -n "${ENABLED_OPERATOR}" ]]; then
  case "${ENABLED_OPERATOR}" in
    Conv) COUNT=64 ;;
    Add) COUNT=8 ;;
    Mul) COUNT=58 ;;
    Sub|Slice|Resize) COUNT=2 ;;
    Div) COUNT=1 ;;
    Sigmoid) COUNT=58 ;;
    MaxPool) COUNT=3 ;;
    Concat) COUNT=17 ;;
    Split) COUNT=8 ;;
    Transpose|Softmax) COUNT=1 ;;
    *) echo "YOLOv8n 不支持筛选算子：${ENABLED_OPERATOR}" >&2; exit 2 ;;
  esac
  ASSIGNMENT="${ENABLED_OPERATOR}=${COUNT}"
  ENABLE_ARGS=(--enabled-ops "${ENABLED_OPERATOR}")
else
  ASSIGNMENT="${ALL_COUNTS}"
  ENABLE_ARGS=()
fi

"${RUNNER}" \
  --model "${MODEL_DIR}/yolov8n.onnx" \
  --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" \
  --data-dir "${MODEL_DIR}/data" \
  "${ENABLE_ARGS[@]}" \
  --expect-rvnpu-counts "${ASSIGNMENT}" \
  --report-max-error-percent \
  --atol 1e-3 --rtol 1e-3
