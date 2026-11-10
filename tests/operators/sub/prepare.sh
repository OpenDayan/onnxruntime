#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
"${PYTHON_EXECUTABLE:-python3}" "${ROOT_DIR}/tests/models/make_elementwise_models.py" \
  --op sub --output-dir "${ROOT_DIR}/build/operators/sub"
