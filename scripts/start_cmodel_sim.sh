#!/usr/bin/env bash
# C-model backend: dayan-emu + custom QEMU + dayan_server.
# No RTL simulator or npu-setup checkout is used. Build artifacts separately.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAYAN_ROOT="${DAYAN_ROOT:-${ROOT_DIR}/../dayan-base}"
DAYAN_EMU_DIR="${DAYAN_EMU_DIR:-${ROOT_DIR}/../dayan-emu}"
DAYAN_EMU_CORES="${DAYAN_EMU_CORES:-4}"
DAYAN_EMU_MEMORY_MIB="${DAYAN_EMU_MEMORY_MIB:-4096}"
DAYAN_SERVER_SOCKET="${DAYAN_SERVER_SOCKET:-/tmp/dayan-server.sock}"
DAYAN_EMU_LOG_DIR="${DAYAN_EMU_LOG_DIR:-${ROOT_DIR}/build/emu}"
QEMU_DL_BIN="${QEMU_DL_BIN:-${DAYAN_EMU_DIR}/deps/npu-qemu/build/dl-riscv64}"
DEVICE_DIR="${DAYAN_XDMA_DEVICE_DIR:-${DAYAN_EMU_LOG_DIR}/virtual_npu}"

if [[ "${1:-}" == --help ]]; then
    cat <<'EOF'
Usage: bash scripts/start_cmodel_sim.sh

Foreground launcher; Ctrl-C stops only the processes it starts.
Build QEMU/emulator, dayan-base firmware/Runtime and the ONNX EP first.

Environment:
  DAYAN_ROOT             dayan-base checkout (default: sibling dayan-base)
  DAYAN_EMU_DIR          emulator checkout (default: sibling dayan-emu)
  QEMU_DL_BIN            matching custom QEMU executable
  DAYAN_EMU_CORES        1..4 (default: 4)
  DAYAN_EMU_MEMORY_MIB   shared DRAM/QEMU/HAL size, MiB (default: 4096)
  DAYAN_SERVER_SOCKET   Runtime socket (default: /tmp/dayan-server.sock)
  DAYAN_EMU_LOG_DIR      logs (default: onnxruntime/build/emu)
  DAYAN_XDMA_DEVICE_DIR  FUSE endpoints (default: LOG_DIR/virtual_npu)

The emulator has fixed global IPC names, despite configurable FUSE endpoints:
do not run it concurrently with another emulator/RTL backend.
This script does not resize /dev/shm or install dependencies.
EOF
    exit 0
fi
[[ $# == 0 ]] || { echo "Unexpected arguments; use --help" >&2; exit 2; }
[[ "$DAYAN_EMU_CORES" =~ ^[1-4]$ ]] ||
    { echo "DAYAN_EMU_CORES must be 1..4" >&2; exit 2; }
[[ "$DAYAN_EMU_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] ||
    { echo "DAYAN_EMU_MEMORY_MIB must be a positive integer" >&2; exit 2; }

EMU="${DAYAN_EMU_DIR}/bin/rvt_emulator"
SERVER="${DAYAN_ROOT}/build/host/driver/dayan_server"
BOOTROM="${DAYAN_ROOT}/device/bl0/bootrom.elf"
FIRMWARE="${DAYAN_ROOT}/build/device/npu-os.elf"
API_ELF="${DAYAN_ROOT}/build/device/npu-os-api.elf"
for file in "$EMU" "$QEMU_DL_BIN" "$SERVER"; do
    [[ -x "$file" ]] || { echo "Missing executable: $file" >&2; exit 2; }
done
for file in "$BOOTROM" "$FIRMWARE" "$API_ELF"; do
    [[ -r "$file" ]] || { echo "Missing firmware: $file" >&2; exit 2; }
done
[[ -r /dev/fuse && -w /dev/fuse ]] ||
    { echo "/dev/fuse is not accessible" >&2; exit 2; }
command -v fusermount3 >/dev/null
command -v setsid >/dev/null
if pgrep -x rvt_emulator >/dev/null || pgrep -x dl-riscv64 >/dev/null ||
   pgrep -x TestCluster >/dev/null || findmnt -rn --mountpoint "$DEVICE_DIR" >/dev/null; then
    echo "An emulator/RTL backend or its FUSE mount is already active; refusing to stop it." >&2
    exit 2
fi

IPC_FILES=(/tmp/card0_rvt_msg /tmp/card0_rvt_inst_req
           /tmp/card0_rvt_inst_rsp /tmp/card0_rvt_mmu /dev/shm/dram_shm_0)
for ((core=0; core<8; core++)); do
    IPC_FILES+=("/dev/shm/uram_shm_${core}")
done
for file in "$DAYAN_SERVER_SOCKET" "${IPC_FILES[@]}"; do
    if [[ -e "$file" || -L "$file" ]]; then
        # The emulator itself unlinks/rebinds its sockets. Allow only an
        # owned, non-symlink socket with no live kernel socket entry.
        if [[ "$file" == /tmp/card0_rvt_* && -S "$file" && -O "$file" &&
              ! -L "$file" ]] &&
           ! awk -v path="$file" '$8 == path { found=1 } END { exit !found }' /proc/net/unix; then
            echo "Emulator may replace its inactive socket: $file" >&2
            continue
        fi
        echo "Existing IPC resource: $file; inspect its owner before removing it." >&2
        exit 2
    fi
done
# Conservative capacity check: DRAM + per-core URAM + 8 MiB spare.
# tmpfs is demand-paged; this checks capacity, not host/cgroup RAM availability.
python3 - "$DAYAN_EMU_MEMORY_MIB" "$DAYAN_EMU_CORES" <<'PY'
import os
import sys
s = os.statvfs("/dev/shm")
available = s.f_bavail * s.f_frsize
needed = int(sys.argv[1]) * 1048576 + int(sys.argv[2]) * 1536 * 1024 + 8 * 1048576
if available < needed:
    sys.exit(f"/dev/shm has {available / 1048576:.1f} MiB available; "
             f"need at least {needed / 1048576:.1f} MiB for this configuration. "
             "Increase the container shared-memory limit, or explicitly choose "
             "a smaller DAYAN_EMU_MEMORY_MIB for a small smoke test.")
PY

mkdir -p "$DAYAN_EMU_LOG_DIR" "$DEVICE_DIR"
emu_pid=
server_pid=
cleanup() {
    local pid
    trap - EXIT INT TERM
    for pid in "$server_pid" "$emu_pid"; do
        [[ -n "$pid" ]] || continue
        # Each process is in a private session; also stop the QEMU child.
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for _ in {1..20}; do
        if ! { [[ -n "$emu_pid" ]] && kill -0 -- "-$emu_pid" 2>/dev/null; } &&
           ! { [[ -n "$server_pid" ]] && kill -0 -- "-$server_pid" 2>/dev/null; }; then
            break
        fi
        sleep 0.1
    done
    for pid in "$server_pid" "$emu_pid"; do
        [[ -n "$pid" ]] || continue
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if [[ -n "$emu_pid" ]]; then
        if findmnt -rn --mountpoint "$DEVICE_DIR" >/dev/null; then
            fusermount3 -uz "$DEVICE_DIR" 2>/dev/null || true
        fi
        rm -f "${IPC_FILES[@]}"
    fi
    [[ -z "$server_pid" ]] || rm -f "$DAYAN_SERVER_SOCKET"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export DAYAN_EMU_MEMORY_MIB
SECONDS=0
NPU_DEVICE_PREFIX="$DEVICE_DIR" setsid "$EMU" -c "$DAYAN_EMU_CORES" "$QEMU_DL_BIN" \
    -monitor null -bios "$BOOTROM" -machine rvtng-v0 \
    -nographic -serial none -m "$DAYAN_EMU_MEMORY_MIB" \
    >"$DAYAN_EMU_LOG_DIR/emulator.log" 2>&1 &
emu_pid=$!
for ((attempt=0; attempt<300; attempt++)); do
    kill -0 "$emu_pid" 2>/dev/null ||
        { echo "Emulator exited; see $DAYAN_EMU_LOG_DIR/emulator.log" >&2; exit 1; }
    [[ -r "$DEVICE_DIR/xdma0_user" && -r "$DEVICE_DIR/xdma0_c2h_0" &&
       -w "$DEVICE_DIR/xdma0_h2c_0" ]] && break
    sleep 0.1
done
[[ -r "$DEVICE_DIR/xdma0_user" && -r "$DEVICE_DIR/xdma0_c2h_0" &&
   -w "$DEVICE_DIR/xdma0_h2c_0" ]] ||
    { echo "Device readiness timed out; see emulator.log" >&2; exit 1; }
setsid "$SERVER" --socket "$DAYAN_SERVER_SOCKET" \
    --firmware "$FIRMWARE" --api-elf "$API_ELF" \
    --xdma-profile simulator --xdma-device-dir "$DEVICE_DIR" \
    --device-index 0 --cluster-count 1 --cores-per-cluster "$DAYAN_EMU_CORES" \
    --core-mask "$(( (1 << DAYAN_EMU_CORES) - 1 ))" \
    --enable-mmu 0 --normal-memory-size-mib "$DAYAN_EMU_MEMORY_MIB" \
    --timeout-ms 60000 >"$DAYAN_EMU_LOG_DIR/server.log" 2>&1 &
server_pid=$!
for ((attempt=0; attempt<600; attempt++)); do
    if ! kill -0 "$emu_pid" 2>/dev/null || ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Backend exited; see $DAYAN_EMU_LOG_DIR/{emulator,server}.log" >&2
        exit 1
    fi
    [[ -S "$DAYAN_SERVER_SOCKET" ]] && break
    sleep 0.1
done
[[ -S "$DAYAN_SERVER_SOCKET" ]] ||
    { echo "Server readiness timed out; see server.log" >&2; exit 1; }
printf 'READY backend=cmodel cores=%s memory_mib=%s startup_seconds=%s socket=%s\n' \
    "$DAYAN_EMU_CORES" "$DAYAN_EMU_MEMORY_MIB" "$SECONDS" "$DAYAN_SERVER_SOCKET"
printf 'Logs: %s; Ctrl-C to stop.\n' "$DAYAN_EMU_LOG_DIR"
while kill -0 "$emu_pid" 2>/dev/null && kill -0 "$server_pid" 2>/dev/null; do
    sleep 0.2
done
echo "Backend exited unexpectedly; see $DAYAN_EMU_LOG_DIR" >&2
exit 1
