#!/usr/bin/env bash
# 启动 TestCluster 仿真器 + dayan_server(先清理旧实例)
set -u
TC=/workspace/npu-cluster-sim
pkill -x dayan_server 2>/dev/null; sleep 1
tmux kill-session -t sim 2>/dev/null; sleep 1
pkill -9 -x TestCluster 2>/dev/null; sleep 3
rm -f /tmp/dayan-server.sock
if mountpoint -q /tmp/virtual_npu 2>/dev/null; then
    fusermount3 -uz /tmp/virtual_npu 2>/dev/null || umount -l /tmp/virtual_npu 2>/dev/null
    sleep 1
fi
tmux new-session -d -s sim -x 210 -y 52 -c "$TC" "./TestCluster"
sleep 15
tmux send-keys -t sim "c" Enter
nohup /workspace/dayan-base/build/host/driver/dayan_server \
    --socket /tmp/dayan-server.sock \
    --firmware /workspace/dayan-base/build/device/npu-os.elf \
    --api-elf /workspace/dayan-base/build/device/npu-os-api.elf \
    --xdma-profile simulator --xdma-device-dir /tmp/virtual_npu \
    --device-index 0 --cluster-count 1 --cluster-stride 1 --core-mask 1 \
    --normal-memory-size-mib 4096 --timeout-ms 18000000 \
    > /tmp/dayan-server.log 2>&1 &
for i in $(seq 1 40); do
    sleep 3
    [ -S /tmp/dayan-server.sock ] && break
done
if [ -S /tmp/dayan-server.sock ]; then
    echo "READY  周期读数: tmux capture-pane -t sim -p | grep -oE \"Cycl[e]*[0-9]+\" | tail -1"
else
    echo "FAIL: server socket 未出现,看 /tmp/dayan-server.log"; exit 1
fi
