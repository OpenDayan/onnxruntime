#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
MODEL_DIR="${ROOT_DIR}/build/models"

mkdir -p "${MODEL_DIR}"
"${PYTHON_EXECUTABLE}" "${ROOT_DIR}/tests/models/make_tiny_cnn.py" --output-dir "${MODEL_DIR}"
