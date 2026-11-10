#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
"${PYTHON_EXECUTABLE:-python3}" "${ROOT_DIR}/tests/models/make_transform_models.py" \
  --op transpose --output-dir "${ROOT_DIR}/build/operators/transpose"
