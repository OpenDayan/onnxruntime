#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/tmp/dummy-kernel-elf}"
YOLO_RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_yolov8n"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"
DATA_DIR="${MODEL_DIR}/data"
if [[ -n "${RVNPU_YOLO_IMAGE:-}" ]]; then
  DATA_DIR="${MODEL_DIR}/image_data"
fi

EXTRA_ARGS=()
if [[ -n "${RVNPU_YOLO_IMAGE:-}" ]]; then
  RESULT_DIR="${MODEL_DIR}/image_results"
  EXTRA_ARGS=(--smoke --results-json "${RESULT_DIR}/results.json")
fi
"${YOLO_RUNNER}" --model "${MODEL_DIR}/yolov8n.onnx" \
  --plugin "${PLUGIN}" --kernel-elf "${ELF}" --data-dir "${DATA_DIR}" \
  "${EXTRA_ARGS[@]}"
if [[ -n "${RVNPU_YOLO_IMAGE:-}" ]]; then
  "${PYTHON_EXECUTABLE:-python3}" "${ROOT_DIR}/tests/yolov8n_image.py" draw \
    --metadata "${DATA_DIR}/image_meta.json" \
    --results "${RESULT_DIR}/results.json" --output-dir "${RESULT_DIR}"
fi
