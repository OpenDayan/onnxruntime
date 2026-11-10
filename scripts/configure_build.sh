#!/usr/bin/env bash
# 一键构建:SDK 准备 + dycc 内核编译 + cmake + 单元测试
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
CMAKE_COMMAND="${CMAKE_COMMAND:-cmake}"

"${ROOT_DIR}/scripts/fetch_onnxruntime_sdk.sh"
"${ROOT_DIR}/scripts/build_dayan_kernels.sh"

"${CMAKE_COMMAND}" -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNXRUNTIME_ROOT="${ROOT_DIR}/build/ort-sdk" \
  -DONNXRUNTIME_LIB_DIR="${ROOT_DIR}/build/ort-sdk/lib"
"${CMAKE_COMMAND}" --build "${BUILD_DIR}" --parallel "${JOBS:-8}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
