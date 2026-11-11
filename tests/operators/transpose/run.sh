#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${DAYAN_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
DATA="${ROOT_DIR}/build/operators/transpose/supported"
echo "[1/2] Direct kernel 数值测试"
"${BUILD_DIR}/test_dayan_transform" --kernel-elf "${ELF}" --op transpose
echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${BUILD_DIR}/test_dayan_onnx_model" --model "${DATA}/model.onnx" \
  --plugin "${BUILD_DIR}/libdayan_ep.so" --kernel-elf "${ELF}" \
  --data-dir "${DATA}" --expect-dayan Transpose
