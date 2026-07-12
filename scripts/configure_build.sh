#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/linux}"
ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:?ONNXRUNTIME_ROOT is required}"
NPU_RUNTIME_ROOT="${NPU_RUNTIME_ROOT:?NPU_RUNTIME_ROOT is required}"
ONNXRUNTIME_LIB_DIR="${ONNXRUNTIME_LIB_DIR:-${ONNXRUNTIME_ROOT}/build/Linux/Release}"
CMAKE_COMMAND="${CMAKE_COMMAND:-cmake}"

"${CMAKE_COMMAND}" -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}" \
  -DONNXRUNTIME_LIB_DIR="${ONNXRUNTIME_LIB_DIR}" \
  -DNPU_RUNTIME_ROOT="${NPU_RUNTIME_ROOT}"
"${CMAKE_COMMAND}" --build "${BUILD_DIR}" --parallel "${JOBS:-8}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
