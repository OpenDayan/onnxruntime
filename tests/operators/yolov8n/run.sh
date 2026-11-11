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
PYTHON="${PYTHON_EXECUTABLE:-python3}"
MODEL_DIR="${ROOT_DIR}/build/yolov8n"
POSTPROCESS="${YOLO_POSTPROCESS:-manual}"   # manual(默认) | official
IMAGE="${DAYAN_YOLO_IMAGE:-${ROOT_DIR}/tests/assets/bus.jpg}"

if [[ "${POSTPROCESS}" == "manual" ]]; then
  MODEL="${MODEL_DIR}/yolov8n_manual.onnx"
else
  MODEL="${MODEL_DIR}/yolov8n.onnx"
fi
name="$(basename "${IMAGE%.*}")"
data_dir="${MODEL_DIR}/images/${name}/data"
result_dir="${MODEL_DIR}/images/${name}_${POSTPROCESS}/results"

echo "========== yolov8n image=${name} postprocess=${POSTPROCESS} =========="
"${BUILD_DIR}/test_dayan_yolov8n" --model "${MODEL}" \
  --plugin "${BUILD_DIR}/libdayan_ep.so" \
  --kernel-elf "${ELF}" --data-dir "${data_dir}" \
  --results-json "${result_dir}/results.json"
"${PYTHON}" "${ROOT_DIR}/tools/yolov8n_image.py" draw \
  --metadata "${data_dir}/image_meta.json" \
  --results "${result_dir}/results.json" --output-dir "${result_dir}"
