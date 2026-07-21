#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
"${PYTHON_EXECUTABLE:-python3}" "${ROOT_DIR}/tests/make_nms_models.py" \
  --output-dir "${ROOT_DIR}/build/operators/nms"
