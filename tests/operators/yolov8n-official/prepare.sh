#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLO_POSTPROCESS=official bash "${ROOT_DIR}/tests/operators/yolov8n/prepare.sh"
