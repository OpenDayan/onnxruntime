#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON_EXECUTABLE:-python3}"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"
MODEL="${MODEL_DIR}/yolov8n.onnx"
PREPARE_MODEL=("${PYTHON}" "${ROOT_DIR}/tests/prepare_yolov8n.py")

mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL}" ]] || \
   ! "${PREPARE_MODEL[@]}" --model "${MODEL}" >/dev/null 2>&1; then
  "${PYTHON}" "${ROOT_DIR}/scripts/export_yolov8n.py" \
    --output-dir "${MODEL_DIR}"
fi
"${PREPARE_MODEL[@]}" --model "${MODEL}"
"${PYTHON}" "${ROOT_DIR}/tests/make_yolov8n_manual.py" \
  --official "${MODEL}" --output "${MODEL_DIR}/yolov8n_manual.onnx"
if [[ -n "${RVNPU_YOLO_IMAGE:-}" ]]; then
  "${PYTHON}" "${ROOT_DIR}/tests/yolov8n_image.py" prepare \
    --model "${MODEL}" --image "${RVNPU_YOLO_IMAGE}" \
    --output-dir "${MODEL_DIR}/image_data"
else
  "${PREPARE_MODEL[@]}" \
    --model "${MODEL}" \
    --output-dir "${MODEL_DIR}/data"
fi
