#!/usr/bin/env bash
# 两网算子矩阵直测:水位>42M 自动重启归零;每 case 记 rc+周期到 CSV
set -u
ORT=/workspace/onnxruntime-rvnpu
TC=/workspace/npu-cluster-sim
BIN=$ORT/build/dayan-migration
export LD_LIBRARY_PATH=$ORT/build/ort-sdk/lib:${LD_LIBRARY_PATH:-}
export DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock
CSV=$ORT/build/operator_matrix/results.csv
[ -f $CSV ] || echo "net,tag,shape,rc,cycles" > $CSV

tc_cycles() {
    tmux capture-pane -t sim -p 2>/dev/null | grep -oE "Cycl[e]*[0-9]+" | tail -1 | tr -d "Cycle"
}

restart_stack() {
    echo "$(date +%H:%M:%S) restart (water)" >> $ORT/build/operator_matrix/run.log
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
    sleep 2
}

# 若 server/socket/tmux 不在,先重启
if ! { [ -S /tmp/dayan-server.sock ] && tmux has-session -t sim 2>/dev/null; }; then
    restart_stack
fi

run_case() {
    local net="$1" tag="$2" tool="$3"; shift 3
    local args="$*"
    local w c0 c1 rc
    if grep -q "^$net,$tag," $CSV 2>/dev/null; then
        return 0
    fi
    w=$(tc_cycles); w=${w:-0}
    case "$tag" in
    C3_*) [ "$w" -gt 3000000 ] && restart_stack ;;
    *) [ "$w" -gt 20000000 ] && restart_stack ;;
    esac
    c0=$(tc_cycles); c0=${c0:-0}
    timeout 21600 "$BIN/$tool" --kernel-elf /tmp/dummy-kernel-elf "$@" \
        > $ORT/build/operator_matrix/last.log 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        restart_stack
        timeout 21600 "$BIN/$tool" --kernel-elf /tmp/dummy-kernel-elf "$@" \
            > $ORT/build/operator_matrix/last.log 2>&1
        rc=$?
    fi
    sleep 1
    c1=$(tc_cycles); c1=${c1:-0}
    local d=$((c1 - c0))
    [ "$d" -lt 0 ] && d=0
    echo "$net,$tag,\"$args\",$rc,$d" >> $CSV
    echo "$(date +%H:%M:%S) $net/$tag rc=$rc cyc=$d water=$c1" >> $ORT/build/operator_matrix/run.log
}

C=$BIN/test_rvnpu_conv_maxpool
E=$BIN/test_rvnpu_elementwise
T=$BIN/test_rvnpu_transform
K=$BIN/test_rvnpu_topk
N=$BIN/test_rvnpu_nms

conv() { run_case "$1" "$2" test_rvnpu_conv_maxpool --op conv --input-shape "$3" --weight-shape "$4" --stride "$5" --pad "$6" ${7:+--$7}; }
pool() { run_case "$1" "$2" test_rvnpu_conv_maxpool --op maxpool --input-shape "$3" --kernel "$4" --stride "$5" --pad "$6"; }

# ============ 64 网:conv 3×3 真实形状(17 组代表) ============
conv y64 c3_1 1,3,64,64 16,3,3,3 2,2 1,1 bias
conv y64 c3_2 1,16,32,32 32,16,3,3 2,2 1,1
conv y64 c3_3 1,16,16,16 16,16,3,3 1,1 1,1
conv y64 c3_4 1,32,16,16 64,32,3,3 2,2 1,1
conv y64 c3_5 1,32,8,8 32,32,3,3 1,1 1,1
conv y64 c3_6 1,64,4,4 64,64,3,3 1,1 1,1
conv y64 c3_7 1,128,4,4 128,128,3,3 2,2 1,1
conv y64 c3_8 1,64,8,8 128,64,3,3 2,2 1,1
conv y64 c3_9 1,64,8,8 64,64,3,3 1,1 1,1
conv y64 c3_10 1,128,4,4 64,128,3,3 1,1 1,1
conv y64 c3_11 1,256,2,2 64,256,3,3 1,1 1,1
conv y64 c3_12 1,128,2,2 128,128,3,3 1,1 1,1
conv y64 c3_13 1,64,8,8 80,64,3,3 1,1 1,1
conv y64 c3_14 1,80,8,8 80,80,3,3 1,1 1,1
conv y64 c3_15 1,128,4,4 80,128,3,3 1,1 1,1
conv y64 c3_16 1,80,4,4 80,80,3,3 1,1 1,1
conv y64 c3_17 1,256,2,2 80,256,3,3 1,1 1,1

# ============ 64 网:conv 1×1 真实形状(9 组代表,②c 路径) ============
conv y64 c1_1 1,32,16,16 32,32,1,1 1,1 0,0
conv y64 c1_2 1,48,16,16 32,48,1,1 1,1 0,0
conv y64 c1_3 1,64,8,8 64,64,1,1 1,1 0,0
conv y64 c1_4 1,128,8,8 64,128,1,1 1,1 0,0
conv y64 c1_5 1,128,4,4 128,128,1,1 1,1 0,0
conv y64 c1_6 1,256,4,4 128,256,1,1 1,1 0,0
conv y64 c1_7 1,256,2,2 256,256,1,1 1,1 0,0
conv y64 c1_8 1,384,2,2 256,384,1,1 1,1 0,0
conv y64 c1_9 1,16,4,84 1,16,1,1 1,1 0,0

# ============ 64 网:elementwise 真实形状 ============
for s in 1,16,32,32 1,32,16,16 1,64,8,8 1,128,4,4 1,256,2,2; do
    t=$(echo "$s" | tr ',' 'x')
    run_case y64 "sig_$t" test_rvnpu_elementwise --op sigmoid --shape "$s"
    run_case y64 "mul_$t" test_rvnpu_elementwise --op mul --shape "$s" --rhs-shape "$s"
done
for s in 1,16,16,16 1,32,8,8 1,64,4,4 1,128,2,2; do
    t=$(echo "$s" | tr ',' 'x')
    run_case y64 "add_$t" test_rvnpu_elementwise --op add --shape "$s" --rhs-shape "$s"
done
run_case y64 div_scalar test_rvnpu_elementwise --op div --shape 1 --rhs-shape 1

# ============ 64 网:其他(工具能力内) ============
pool y64 mp_128x2x2 1,128,2,2 5,5 1,1 2,2
run_case y64 softmax_fix test_rvnpu_transform --op softmax
run_case y64 reduce_fix test_rvnpu_transform --op reduce_max
run_case y64 transpose_fix test_rvnpu_transform --op transpose
run_case y64 resize_fix test_rvnpu_transform --op resize
run_case y64 argmax_84x80 test_rvnpu_topk --rows 80 --columns 84 --k 1
run_case y64 nms_case test_rvnpu_nms --case nonoverlap

# ============ 640 网:conv 1×1 真实形状(13 组代表) ============
conv y640 C1_1 1,32,160,160 32,32,1,1 1,1 0,0
conv y640 C1_2 1,48,160,160 32,48,1,1 1,1 0,0
conv y640 C1_3 1,64,80,80 64,64,1,1 1,1 0,0
conv y640 C1_4 1,128,80,80 64,128,1,1 1,1 0,0
conv y640 C1_5 1,128,40,40 128,128,1,1 1,1 0,0
conv y640 C1_6 1,256,40,40 128,256,1,1 1,1 0,0
conv y640 C1_7 1,256,20,20 256,256,1,1 1,1 0,0
conv y640 C1_8 1,512,20,20 256,512,1,1 1,1 0,0
conv y640 C1_9 1,384,20,20 256,384,1,1 1,1 0,0
conv y640 C1_10 1,192,40,40 128,192,1,1 1,1 0,0
conv y640 C1_11 1,192,80,80 64,192,1,1 1,1 0,0
conv y640 C1_12 1,96,80,80 64,96,1,1 1,1 0,0
conv y640 C1_13 1,16,4,8400 1,16,1,1 1,1 0,0

# ============ 640 网:conv 3×3(≤45M 目标,原尺寸或等比缩小) ============
conv y640 C3_min1 1,128,20,20 128,128,3,3 1,1 1,1
conv y640 C3_min2 1,256,12,12 64,256,3,3 1,1 1,1
conv y640 C3_64ch 1,64,26,26 64,64,3,3 1,1 1,1
conv y640 C3_80ch 1,80,25,25 80,80,3,3 1,1 1,1
conv y640 C3_32ch 1,32,37,37 32,32,3,3 1,1 1,1
conv y640 C3_16ch 1,16,36,36 16,16,3,3 1,1 1,1
conv y640 C3_s2_1 1,16,110,110 32,16,3,3 2,2 1,1
conv y640 C3_s2_2 1,32,36,36 64,32,3,3 2,2 1,1
conv y640 C3_s2_3 1,64,54,54 128,64,3,3 2,2 1,1
conv y640 C3_s2_4 1,128,38,38 256,128,3,3 2,2 1,1
conv y640 C3_s2_5 1,3,258,258 16,3,3,3 2,2 1,1 bias
conv y640 C3_128deep 1,128,19,19 128,128,3,3 1,1 1,1
conv y640 C3_256deep 1,256,12,12 80,256,3,3 1,1 1,1

# ============ 640 网:elementwise(原尺寸可测 + 缩放代表) ============
for s in 1,128,20,20 1,80,20,20 1,64,20,20; do
    t=$(echo "$s" | tr ',' 'x')
    run_case y640 "sig_$t" test_rvnpu_elementwise --op sigmoid --shape "$s"
    run_case y640 "mul_$t" test_rvnpu_elementwise --op mul --shape "$s" --rhs-shape "$s"
done
run_case y640 sig_80x8400 test_rvnpu_elementwise --op sigmoid --shape 1,80,8400
run_case y640 sig_scaled_16ch test_rvnpu_elementwise --op sigmoid --shape 1,16,250,250
run_case y640 sig_scaled_32ch test_rvnpu_elementwise --op sigmoid --shape 1,32,176,176
run_case y640 sig_scaled_64ch test_rvnpu_elementwise --op sigmoid --shape 1,64,125,125
run_case y640 sig_scaled_128ch test_rvnpu_elementwise --op sigmoid --shape 1,128,88,88
run_case y640 sig_scaled_256ch test_rvnpu_elementwise --op sigmoid --shape 1,256,62,62
run_case y640 mul_scaled_64ch test_rvnpu_elementwise --op mul --shape 1,64,125,125 --rhs-shape 1,64,125,125
for s in 1,16,160,160 1,32,80,80 1,64,40,40 1,128,20,20; do
    t=$(echo "$s" | tr ',' 'x')
    run_case y640 "add_$t" test_rvnpu_elementwise --op add --shape "$s" --rhs-shape "$s"
done
run_case y640 div_scalar test_rvnpu_elementwise --op div --shape 1 --rhs-shape 1

# ============ 640 网:其他 ============
pool y640 MP_scaled 1,128,15,15 5,5 1,1 2,2
run_case y640 argmax_8400x80 test_rvnpu_topk --rows 80 --columns 8400 --k 1
run_case y640 topk_8400x80 test_rvnpu_topk --rows 80 --columns 8400 --k 300
run_case y640 nms_case test_rvnpu_nms --case nonoverlap

run_case y64 b_sig_1161616 test_rvnpu_elementwise --op sigmoid --shape "1,16,16,16"
run_case y64 b_mul_1161616 test_rvnpu_elementwise --op mul --shape "1,16,16,16" --rhs-shape "1,16,16,16"
run_case y64 b_sig_13288 test_rvnpu_elementwise --op sigmoid --shape "1,32,8,8"
run_case y64 b_mul_13288 test_rvnpu_elementwise --op mul --shape "1,32,8,8" --rhs-shape "1,32,8,8"
run_case y64 b_sig_16444 test_rvnpu_elementwise --op sigmoid --shape "1,64,4,4"
run_case y64 b_mul_16444 test_rvnpu_elementwise --op mul --shape "1,64,4,4" --rhs-shape "1,64,4,4"
conv y64 b_y64_11284425612833 1,128,4,4 256,128,3,3 2,2 1,1
run_case y64 b_sig_112822 test_rvnpu_elementwise --op sigmoid --shape "1,128,2,2"
run_case y64 b_mul_112822 test_rvnpu_elementwise --op mul --shape "1,128,2,2" --rhs-shape "1,128,2,2"
conv y64 b_y64_12562212825611 1,256,2,2 128,256,1,1 1,1 0,0
conv y64 b_y64_15122225651211 1,512,2,2 256,512,1,1 1,1 0,0
conv y64 b_y64_13844412838411 1,384,4,4 128,384,1,1 1,1 0,0
conv y64 b_y64_11924412819211 1,192,4,4 128,192,1,1 1,1 0,0
conv y64 b_y64_11928864192111 1,192,8,8 64,192,1,1 1,1 0,0
conv y64 b_y64_1968864961111 1,96,8,8 64,96,1,1 1,1 0,0
conv y64 b_y64_1648864643322 1,64,8,8 64,64,3,3 2,2 1,1
conv y64 b_y64_1644464641111 1,64,4,4 64,64,1,1 1,1 0,0
run_case y64 b_sig_16422 test_rvnpu_elementwise --op sigmoid --shape "1,64,2,2"
run_case y64 b_mul_16422 test_rvnpu_elementwise --op mul --shape "1,64,2,2" --rhs-shape "1,64,2,2"
conv y64 b_y64_1642264643311 1,64,2,2 64,64,3,3 1,1 1,1
conv y64 b_y64_1642264641111 1,64,2,2 64,64,1,1 1,1 0,0
run_case y64 b_sig_18088 test_rvnpu_elementwise --op sigmoid --shape "1,80,8,8"
run_case y64 b_mul_18088 test_rvnpu_elementwise --op mul --shape "1,80,8,8" --rhs-shape "1,80,8,8"
conv y64 b_y64_1808880801111 1,80,8,8 80,80,1,1 1,1 0,0
run_case y64 b_sig_18044 test_rvnpu_elementwise --op sigmoid --shape "1,80,4,4"
run_case y64 b_mul_18044 test_rvnpu_elementwise --op mul --shape "1,80,4,4" --rhs-shape "1,80,4,4"
conv y64 b_y64_1804480801111 1,80,4,4 80,80,1,1 1,1 0,0
run_case y64 b_sig_18022 test_rvnpu_elementwise --op sigmoid --shape "1,80,2,2"
run_case y64 b_mul_18022 test_rvnpu_elementwise --op mul --shape "1,80,2,2" --rhs-shape "1,80,2,2"
conv y64 b_y64_1802280803311 1,80,2,2 80,80,3,3 1,1 1,1
conv y64 b_y64_1802280801111 1,80,2,2 80,80,1,1 1,1 0,0
run_case y64 b_add_1 test_rvnpu_elementwise --op add --shape "1" --rhs-shape "1"
run_case y64 b_mul_1 test_rvnpu_elementwise --op mul --shape "1" --rhs-shape "1"
run_case y64 b_sig_18084 test_rvnpu_elementwise --op sigmoid --shape "1,80,84"
conv y640 b_y640_13640640163332 1,3,640,640 16,3,3,3 2,2 1,1bias
run_case y640 b_sig_116320320 test_rvnpu_elementwise --op sigmoid --shape "1,16,320,320"
run_case y640 b_mul_116320320 test_rvnpu_elementwise --op mul --shape "1,16,320,320" --rhs-shape "1,16,320,320"
conv y640 b_y640_11632032032163 1,16,320,320 32,16,3,3 2,2 1,1
run_case y640 b_sig_132160160 test_rvnpu_elementwise --op sigmoid --shape "1,32,160,160"
run_case y640 b_mul_132160160 test_rvnpu_elementwise --op mul --shape "1,32,160,160" --rhs-shape "1,32,160,160"
conv y640 b_y640_11616016016163 1,16,160,160 16,16,3,3 1,1 1,1
run_case y640 b_sig_116160160 test_rvnpu_elementwise --op sigmoid --shape "1,16,160,160"
run_case y640 b_mul_116160160 test_rvnpu_elementwise --op mul --shape "1,16,160,160" --rhs-shape "1,16,160,160"
run_case y640 b_add_116160160 test_rvnpu_elementwise --op add --shape "1,16,160,160" --rhs-shape "1,16,160,160"
conv y640 b_y640_13216016064323 1,32,160,160 64,32,3,3 2,2 1,1
run_case y640 b_sig_1648080 test_rvnpu_elementwise --op sigmoid --shape "1,64,80,80"
run_case y640 b_mul_1648080 test_rvnpu_elementwise --op mul --shape "1,64,80,80" --rhs-shape "1,64,80,80"
conv y640 b_y640_13280803232331 1,32,80,80 32,32,3,3 1,1 1,1
run_case y640 b_sig_1328080 test_rvnpu_elementwise --op sigmoid --shape "1,32,80,80"
run_case y640 b_mul_1328080 test_rvnpu_elementwise --op mul --shape "1,32,80,80" --rhs-shape "1,32,80,80"
run_case y640 b_add_1328080 test_rvnpu_elementwise --op add --shape "1,32,80,80" --rhs-shape "1,32,80,80"
conv y640 b_y640_16480801286433 1,64,80,80 128,64,3,3 2,2 1,1
run_case y640 b_sig_11284040 test_rvnpu_elementwise --op sigmoid --shape "1,128,40,40"
run_case y640 b_mul_11284040 test_rvnpu_elementwise --op mul --shape "1,128,40,40" --rhs-shape "1,128,40,40"
conv y640 b_y640_16440406464331 1,64,40,40 64,64,3,3 1,1 1,1
run_case y640 b_sig_1644040 test_rvnpu_elementwise --op sigmoid --shape "1,64,40,40"
run_case y640 b_mul_1644040 test_rvnpu_elementwise --op mul --shape "1,64,40,40" --rhs-shape "1,64,40,40"
run_case y640 b_add_1644040 test_rvnpu_elementwise --op add --shape "1,64,40,40" --rhs-shape "1,64,40,40"
conv y640 b_y640_11284040256128 1,128,40,40 256,128,3,3 2,2 1,1
run_case y640 b_sig_12562020 test_rvnpu_elementwise --op sigmoid --shape "1,256,20,20"
run_case y640 b_mul_12562020 test_rvnpu_elementwise --op mul --shape "1,256,20,20" --rhs-shape "1,256,20,20"
run_case y640 b_sig_11282020 test_rvnpu_elementwise --op sigmoid --shape "1,128,20,20"
run_case y640 b_mul_11282020 test_rvnpu_elementwise --op mul --shape "1,128,20,20" --rhs-shape "1,128,20,20"
run_case y640 b_add_11282020 test_rvnpu_elementwise --op add --shape "1,128,20,20" --rhs-shape "1,128,20,20"
conv y640 b_y640_12562020128256 1,256,20,20 128,256,1,1 1,1 0,0
conv y640 b_y640_13844040128384 1,384,40,40 128,384,1,1 1,1 0,0
conv y640 b_y640_16480806464332 1,64,80,80 64,64,3,3 2,2 1,1
conv y640 b_y640_11284040128128 1,128,40,40 128,128,3,3 2,2 1,1
conv y640 b_y640_16480806464331 1,64,80,80 64,64,3,3 1,1 1,1
conv y640 b_y640_11284040641283 1,128,40,40 64,128,3,3 1,1 1,1
conv y640 b_y640_16440406464111 1,64,40,40 64,64,1,1 1,1 0,0
conv y640 b_y640_12562020642563 1,256,20,20 64,256,3,3 1,1 1,1
run_case y640 b_sig_1642020 test_rvnpu_elementwise --op sigmoid --shape "1,64,20,20"
run_case y640 b_mul_1642020 test_rvnpu_elementwise --op mul --shape "1,64,20,20" --rhs-shape "1,64,20,20"
conv y640 b_y640_16420206464331 1,64,20,20 64,64,3,3 1,1 1,1
conv y640 b_y640_16420206464111 1,64,20,20 64,64,1,1 1,1 0,0
conv y640 b_y640_16480808064331 1,64,80,80 80,64,3,3 1,1 1,1
run_case y640 b_sig_1808080 test_rvnpu_elementwise --op sigmoid --shape "1,80,80,80"
run_case y640 b_mul_1808080 test_rvnpu_elementwise --op mul --shape "1,80,80,80" --rhs-shape "1,80,80,80"
conv y640 b_y640_18080808080331 1,80,80,80 80,80,3,3 1,1 1,1
conv y640 b_y640_18080808080111 1,80,80,80 80,80,1,1 1,1 0,0
conv y640 b_y640_11284040801283 1,128,40,40 80,128,3,3 1,1 1,1
run_case y640 b_sig_1804040 test_rvnpu_elementwise --op sigmoid --shape "1,80,40,40"
run_case y640 b_mul_1804040 test_rvnpu_elementwise --op mul --shape "1,80,40,40" --rhs-shape "1,80,40,40"
conv y640 b_y640_18040408080331 1,80,40,40 80,80,3,3 1,1 1,1
conv y640 b_y640_18040408080111 1,80,40,40 80,80,1,1 1,1 0,0
conv y640 b_y640_12562020802563 1,256,20,20 80,256,3,3 1,1 1,1
run_case y640 b_sig_1802020 test_rvnpu_elementwise --op sigmoid --shape "1,80,20,20"
run_case y640 b_mul_1802020 test_rvnpu_elementwise --op mul --shape "1,80,20,20" --rhs-shape "1,80,20,20"
conv y640 b_y640_18020208080331 1,80,20,20 80,80,3,3 1,1 1,1
conv y640 b_y640_18020208080111 1,80,20,20 80,80,1,1 1,1 0,0
run_case y640 b_sig_1808400 test_rvnpu_elementwise --op sigmoid --shape "1,80,8400"

echo "$(date +%H:%M:%S) ALL DONE" >> $ORT/build/operator_matrix/run.log
