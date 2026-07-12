#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 0 ]]; then
  echo "用法：run_e2e.sh" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d /tmp/rvnpu-e2e.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT
STAGE=0

run_stage() {
  local label="$1"
  shift
  STAGE=$((STAGE + 1))
  local log="${TMP_DIR}/${STAGE}.log"
  printf '[%d/3] %s ... ' "${STAGE}" "${label}"
  local rc=0
  if "$@" >"${log}" 2>&1; then
    echo "PASS"
    return 0
  else
    rc=$?
  fi
  echo "FAIL"
  tail -n 80 "${log}" >&2 || true
  return "${rc}"
}

run_stage "构建 Plugin" "${ROOT_DIR}/scripts/configure_build.sh"
run_stage "构建 device ELF" "${ROOT_DIR}/scripts/build_device_kernel.sh"
run_stage "验证 MatMul" "${ROOT_DIR}/scripts/test_operator.sh" matmul

echo "全部验证通过。"
