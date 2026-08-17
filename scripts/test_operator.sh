#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPERATOR_NAME="${1:-}"
IMAGE_PATH="${2:-}"

if [[ -z "${OPERATOR_NAME}" || $# -gt 2 ]]; then
  echo "用法：test_operator.sh <算子名> [图片路径]" >&2
  exit 2
fi
if [[ -n "${IMAGE_PATH}" && "${OPERATOR_NAME}" != "yolov8n" && \
      "${OPERATOR_NAME}" != "yolov8n_manual" ]]; then
  echo "只有 yolov8n 测试可以指定图片路径" >&2
  exit 2
fi
if [[ -n "${IMAGE_PATH}" && ! -f "${IMAGE_PATH}" ]]; then
  echo "图片不存在：${IMAGE_PATH}" >&2
  exit 2
fi
if [[ "${OPERATOR_NAME}" == "yolov8n" || \
      "${OPERATOR_NAME}" == "yolov8n_manual" ]]; then
  unset RVNPU_YOLO_IMAGE
  if [[ -n "${IMAGE_PATH}" ]]; then
    export RVNPU_YOLO_IMAGE="${IMAGE_PATH}"
  fi
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
