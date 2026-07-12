#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVNPU_COMPILER_ROOT="${RVNPU_COMPILER_ROOT:?RVNPU_COMPILER_ROOT is required}"
NPU_KERNELS_ROOT="${NPU_KERNELS_ROOT:?NPU_KERNELS_ROOT is required}"
NPU_SETUP_ROOT="${NPU_SETUP_ROOT:?NPU_SETUP_ROOT is required}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:?LLVM_BUILD_DIR is required}"

QEMU_LOG="$(mktemp /tmp/rvnpu-device-build.XXXXXX.log)"
trap 'rm -f "${QEMU_LOG}"' EXIT

RVTNG_QEMU_LOG="${QEMU_LOG}" bash "${RVNPU_COMPILER_ROOT}/integration/scripts/run_user_triton_matmul_end2end.sh" \
  --repo-root "${RVNPU_COMPILER_ROOT}" \
  --npu-setup-root "${NPU_SETUP_ROOT}" \
  --npu-kernels-root "${NPU_KERNELS_ROOT}" \
  --llvm-build-dir "${LLVM_BUILD_DIR}" \
  --out-dir "${ROOT_DIR}/build/device" \
  --jobs "${JOBS:-8}"
