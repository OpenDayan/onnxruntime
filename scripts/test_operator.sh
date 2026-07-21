#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPERATOR_NAME="${1:-}"

if [[ -z "${OPERATOR_NAME}" || $# -ne 1 ]]; then
  echo "用法：test_operator.sh <算子名>" >&2
  exit 2
fi

SUITE_DIR="${ROOT_DIR}/tests/operators/${OPERATOR_NAME}"
PREPARE_SCRIPT="${SUITE_DIR}/prepare.sh"
RUN_SCRIPT="${SUITE_DIR}/run.sh"
for required in "${PREPARE_SCRIPT}" "${RUN_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "缺少算子测试文件：${required}" >&2
    exit 2
  fi
done

export RVNPU_KERNEL_ELF="${RVNPU_KERNEL_ELF:-${ROOT_DIR}/build/device/device_build/rvnpu_kernels.elf}"
if [[ ! -f "${RVNPU_KERNEL_ELF}" ]]; then
  echo "device ELF 不存在：${RVNPU_KERNEL_ELF}" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${ONNXRUNTIME_LIB_DIR:-${ONNXRUNTIME_ROOT:?ONNXRUNTIME_ROOT is required}/build/Linux/Release}:${LD_LIBRARY_PATH:-}"

PREPARE_LOG="$(mktemp /tmp/rvnpu-${OPERATOR_NAME}-prepare.XXXXXX.log)"
trap 'rm -f "${PREPARE_LOG}"' EXIT
if ! bash "${PREPARE_SCRIPT}" >"${PREPARE_LOG}" 2>&1; then
  cat "${PREPARE_LOG}" >&2
  exit 1
fi

echo "========== ${OPERATOR_NAME} 算子测试 =========="
"${ROOT_DIR}/scripts/run_with_emulator.sh" -- bash "${RUN_SCRIPT}"
echo "========== ${OPERATOR_NAME} PASS =========="
