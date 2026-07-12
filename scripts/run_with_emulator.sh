#!/usr/bin/env bash
set -euo pipefail

NPU_SETUP_ROOT="${NPU_SETUP_ROOT:?NPU_SETUP_ROOT is required}"
KERNEL_ELF="${RVNPU_KERNEL_ELF:?RVNPU_KERNEL_ELF is required}"
REMOVE_EMU_LOG=0
if [[ -z "${EMU_LOG:-}" ]]; then
  EMU_LOG="$(mktemp /tmp/rvnpu-emulator.XXXXXX.log)"
  REMOVE_EMU_LOG=1
fi
if [[ "${1:-}" != "--" ]]; then
  echo "usage: run_with_emulator.sh -- command [args...]" >&2
  exit 2
fi
shift
if [[ $# -eq 0 ]]; then echo "missing command" >&2; exit 2; fi

first_existing() {
  local value
  for value in "$@"; do
    if [[ -e "${value}" ]]; then echo "${value}"; return 0; fi
  done
  return 1
}

EMU_BIN="$(first_existing "${NPU_SETUP_ROOT}/workspace/rvtng-cmodel/bin/rvt_emulator" "${NPU_SETUP_ROOT}/bin/rvt_emulator")"
DL_BIN="$(first_existing "${NPU_SETUP_ROOT}/workspace/npu-qemu/build/dl-riscv64" "${NPU_SETUP_ROOT}/bin/dl-riscv64")"
BOOTROM_BIN="$(first_existing "${NPU_SETUP_ROOT}/bin/bootrom.elf" "${NPU_SETUP_ROOT}/build/bootrom.elf")"

unmount_virtual_npu() {
  fusermount3 -u /tmp/virtual_npu >/dev/null 2>&1 ||
    umount -l /tmp/virtual_npu >/dev/null 2>&1 || true
}

remove_shared_memory() {
  local shm_files=()
  local path owner
  shopt -s nullglob
  shm_files=(/dev/shm/dram_shm_* /dev/shm/uram_shm_*)
  shopt -u nullglob
  for path in "${shm_files[@]}"; do
    if [[ ${EUID} -ne 0 && ! -O "${path}" ]]; then
      owner="$(stat -c '%U:%G' "${path}" 2>/dev/null || echo unknown)"
      echo "无法清理 ${path}（所有者 ${owner}）。请始终使用同一个容器用户运行测试。" >&2
      return 1
    fi
    rm -f "${path}"
  done
}

cleanup() {
  set +e
  if [[ -n "${EMU_PID:-}" ]]; then
    kill "${EMU_PID}" >/dev/null 2>&1
    wait "${EMU_PID}" >/dev/null 2>&1
  fi
  pkill -f '/bin/rvt_emulator' >/dev/null 2>&1 || true
  pkill -f '/bin/dl-riscv64' >/dev/null 2>&1 || true
  unmount_virtual_npu
  rm -rf /tmp/virtual_npu >/dev/null 2>&1 || true
  remove_shared_memory >/dev/null 2>&1 || true
  if [[ ${REMOVE_EMU_LOG} -eq 1 ]]; then
    rm -f "${EMU_LOG}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "$(dirname "${EMU_LOG}")"
if ! : >"${EMU_LOG}"; then
  echo "模拟器日志不可写：${EMU_LOG}" >&2
  exit 2
fi

pkill -f '/bin/rvt_emulator' >/dev/null 2>&1 || true
pkill -f '/bin/dl-riscv64' >/dev/null 2>&1 || true
unmount_virtual_npu
rm -rf /tmp/virtual_npu
rm -f /tmp/card0_rvt_msg /tmp/card0_rvt_inst /tmp/card0_rvt_inst_req /tmp/card0_rvt_inst_rsp
remove_shared_memory
mkdir -p /tmp/virtual_npu

RVTNG_QEMU_LOG="${EMU_LOG}" "${EMU_BIN}" "${DL_BIN}" -monitor null -bios "${BOOTROM_BIN}" -kernel "${KERNEL_ELF}" \
  -machine rvtng-v0 -m 4096 -nographic -serial none >"${EMU_LOG}" 2>&1 &
EMU_PID=$!
for _ in {1..120}; do
  if [[ -e /tmp/virtual_npu/xdma0_h2c_0 && -e /tmp/virtual_npu/xdma0_c2h_0 ]]; then break; fi
  sleep 0.25
done
if [[ ! -e /tmp/virtual_npu/xdma0_h2c_0 || ! -e /tmp/virtual_npu/xdma0_c2h_0 ]]; then
  echo "emulator device nodes did not appear" >&2
  tail -n 80 "${EMU_LOG}" >&2 || true
  exit 2
fi
"$@"
