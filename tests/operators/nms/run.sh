#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${RVNPU_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
DIRECT_TEST="${BUILD_DIR}/test_rvnpu_nms"
RUNNER="${BUILD_DIR}/test_rvnpu_nms_onnx"
PLUGIN="${BUILD_DIR}/librvnpu_ep.so"
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
"${BUILD_DIR}/test_rvnpu_yolo_postprocess" \
  "${DATA}/yolo_postprocess/model.onnx" "${PLUGIN}" "${ELF}" \
  "${DATA}/yolo_postprocess"
