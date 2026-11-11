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
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${DAYAN_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
DATA="${ROOT_DIR}/build/operators/reduce_max/supported"
echo "[1/2] Direct kernel 数值测试"
"${BUILD_DIR}/test_dayan_transform" --kernel-elf "${ELF}" --op reduce_max
echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${BUILD_DIR}/test_dayan_onnx_model" --model "${DATA}/model.onnx" \
  --plugin "${BUILD_DIR}/libdayan_ep.so" --kernel-elf "${ELF}" \
  --data-dir "${DATA}" --expect-dayan ReduceMax
