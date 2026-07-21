#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON_EXECUTABLE:-python3}"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"
MODEL="${MODEL_DIR}/yolov8n.onnx"

if [[ ! -f "${MODEL}" ]]; then
  "${PYTHON}" "${ROOT_DIR}/scripts/export_yolov8n.py" \
    --output-dir "${MODEL_DIR}"
fi
"${PYTHON}" "${ROOT_DIR}/tests/prepare_yolov8n.py" \
  --model "${MODEL}" --output-dir "${MODEL_DIR}/data"
