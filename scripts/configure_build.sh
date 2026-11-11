#!/usr/bin/env bash
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# onnxruntime is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
CMAKE_COMMAND="${CMAKE_COMMAND:-cmake}"

"${ROOT_DIR}/scripts/fetch_onnxruntime_sdk.sh"
"${ROOT_DIR}/scripts/build_dayan_kernels.sh"

"${CMAKE_COMMAND}" -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDAYAN_ROOT="${DAYAN_ROOT:?set DAYAN_ROOT to the dayan-base repository root}" \
  -DONNXRUNTIME_ROOT="${ROOT_DIR}/build/ort-sdk" \
  -DONNXRUNTIME_LIB_DIR="${ROOT_DIR}/build/ort-sdk/lib"
"${CMAKE_COMMAND}" --build "${BUILD_DIR}" --parallel "${JOBS:-8}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
