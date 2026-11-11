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
DIRECT_TEST="${BUILD_DIR}/test_dayan_nms"
RUNNER="${BUILD_DIR}/test_dayan_nms_onnx"
PLUGIN="${BUILD_DIR}/libdayan_ep.so"
DATA="${ROOT_DIR}/build/operators/nms"

echo "[1/2] Direct kernel 数值测试"
for case_name in nonoverlap overlap multiclass empty boundary center gathernd; do
  "${DIRECT_TEST}" --kernel-elf "${ELF}" --case "${case_name}"
done

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
echo "model=NMS"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported"
echo "model=NMS center_point_box=1"
"${RUNNER}" --model "${DATA}/supported_center/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported_center"
echo "model=TopK+NMS"
"${RUNNER}" --model "${DATA}/topk_pipeline/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/topk_pipeline" \
  --expect-topk
echo "model=YOLO Split+Transpose+NMS+GatherND"
"${BUILD_DIR}/test_dayan_yolo_postprocess" \
  "${DATA}/yolo_postprocess/model.onnx" "${PLUGIN}" "${ELF}" \
  "${DATA}/yolo_postprocess"
