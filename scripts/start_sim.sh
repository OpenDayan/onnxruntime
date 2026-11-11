#!/usr/bin/env bash
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# onnxruntime is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

set -u
: "${RTLSIM_ROOT:?set RTLSIM_ROOT to the npu-cluster-sim repository root}"
: "${DAYAN_ROOT:?set DAYAN_ROOT to the dayan-base repository root}"
TC_SESSION="${DAYAN_TC_SESSION:-sim}"
DAYAN_SERVER_SOCKET="${DAYAN_SERVER_SOCKET:-${TMPDIR:-/tmp}/dayan-server.sock}"
DAYAN_SERVER_LOG="${DAYAN_SERVER_LOG:-${TMPDIR:-/tmp}/dayan-server.log}"
DAYAN_XDMA_DEVICE_DIR="${DAYAN_XDMA_DEVICE_DIR:-${TMPDIR:-/tmp}/virtual_npu}"
pkill -x dayan_server 2>/dev/null; sleep 1
tmux kill-session -t "${TC_SESSION}" 2>/dev/null; sleep 1
pkill -9 -x TestCluster 2>/dev/null; sleep 3
rm -f "${DAYAN_SERVER_SOCKET}"
if mountpoint -q "${DAYAN_XDMA_DEVICE_DIR}" 2>/dev/null; then
    fusermount3 -uz "${DAYAN_XDMA_DEVICE_DIR}" 2>/dev/null || umount -l "${DAYAN_XDMA_DEVICE_DIR}" 2>/dev/null
    sleep 1
fi
tmux new-session -d -s "${TC_SESSION}" -x 210 -y 52 -c "${RTLSIM_ROOT}" "./TestCluster"
sleep 15
tmux send-keys -t "${TC_SESSION}" "c" Enter
nohup "${DAYAN_ROOT}/build/host/driver/dayan_server" \
    --socket "${DAYAN_SERVER_SOCKET}" \
    --firmware "${DAYAN_ROOT}/build/device/npu-os.elf" \
    --api-elf "${DAYAN_ROOT}/build/device/npu-os-api.elf" \
    --xdma-profile simulator --xdma-device-dir "${DAYAN_XDMA_DEVICE_DIR}" \
    --device-index 0 --cluster-count 1 --cluster-stride 1 --core-mask 1 \
    --normal-memory-size-mib 4096 --timeout-ms 18000000 \
    > "${DAYAN_SERVER_LOG}" 2>&1 &
for i in $(seq 1 40); do
    sleep 3
    [ -S "${DAYAN_SERVER_SOCKET}" ] && break
done
if [ -S "${DAYAN_SERVER_SOCKET}" ]; then
    echo "READY  周期读数: tmux capture-pane -t ${TC_SESSION} -p | grep -oE \"Cycl[e]*[0-9]+\" | tail -1"
else
    echo "FAIL: server socket 未出现,看 ${DAYAN_SERVER_LOG}"; exit 1
fi
