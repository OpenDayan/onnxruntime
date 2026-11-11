#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
ELF="${DAYAN_KERNEL_ELF:-${BUILD_DIR}/dummy-kernel-elf}"
[[ -e "${ELF}" ]] || : > "${ELF}"
DIRECT_TEST="${BUILD_DIR}/test_dayan_elementwise"
RUNNER="${BUILD_DIR}/test_dayan_onnx_model"
PLUGIN="${BUILD_DIR}/libdayan_ep.so"
DATA="${ROOT_DIR}/build/operators/sigmoid"

echo "[1/2] Direct kernel 数值测试"
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op sigmoid --shape 2,3,5
"${DIRECT_TEST}" --kernel-elf "${ELF}" --op sigmoid --shape 1,2,3,16

echo
echo "[2/2] ONNX Runtime 分配与数值测试"
"${RUNNER}" --model "${DATA}/supported/model.onnx" --plugin "${PLUGIN}" \
  --kernel-elf "${ELF}" --data-dir "${DATA}/supported" \
  --expect-dayan Sigmoid --atol 1e-4 --rtol 1e-4
