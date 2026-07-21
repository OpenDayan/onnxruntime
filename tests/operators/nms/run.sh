#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:?RVNPU_KERNEL_ELF is required}"
DIRECT_TEST="${ROOT_DIR}/build/linux/test_rvnpu_nms"
RUNNER="${ROOT_DIR}/build/linux/test_rvnpu_nms_onnx"
PLUGIN="${ROOT_DIR}/build/linux/librvnpu_ep.so"
DATA="${ROOT_DIR}/build/operators/nms"

echo "[1/2] Direct kernel 数值测试"
for case_name in nonoverlap overlap multiclass empty boundary; do
  "${DIRECT_TEST}" --kernel-elf "${ELF}" --case "${case_name}"
done

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
echo "model=NMS"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported"
echo "model=TopK+NMS"
"${RUNNER}" --model "${DATA}/topk_pipeline/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/topk_pipeline" \
  --expect-topk
