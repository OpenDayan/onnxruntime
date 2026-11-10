#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON_EXECUTABLE:-python3}"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"
MODEL="${MODEL_DIR}/yolov8n.onnx"
# 换测试图:改这里的路径,或运行时设 RVNPU_YOLO_IMAGE
IMAGE="${RVNPU_YOLO_IMAGE:-${ROOT_DIR}/tests/assets/bus.jpg}"

mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL}" ]]; then
  "${PYTHON}" "${ROOT_DIR}/tools/export_yolov8n.py" \
    --output-dir "${MODEL_DIR}"
fi
"${PYTHON}" "${ROOT_DIR}/tests/models/make_yolov8n_manual.py" \
  --official "${MODEL}" --output "${MODEL_DIR}/yolov8n_manual.onnx"

name="$(basename "${IMAGE%.*}")"
"${PYTHON}" "${ROOT_DIR}/tools/yolov8n_image.py" prepare \
  --model "${MODEL}" --image "${IMAGE}" \
  --output-dir "${MODEL_DIR}/images/${name}/data"
