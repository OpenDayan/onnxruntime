# RVNPU ONNX Runtime Plugin EP

本项目提供独立的 `RVNPUExecutionProvider`,不修改官方 ONNX Runtime。

当前支持标准 ONNX 算子:

```text
MatMul, Add, Mul, Sub, Div, Sigmoid, Conv, MaxPool,
Concat, Split, Slice, Resize, Transpose, Softmax, ReduceMax, TopK,
ArgMax, NonMaxSuppression, GatherND
```

设备内核为 `/workspace/dayan-base/demo/onnx/kernel_*.cpp`(含 conv 的
im2col+matmul、elementwise、softmax 等),经 dycc 编译为 `.o` 后链接进 EP。

## 1. 环境要求

| 仓库 | 分支 | 用途 |
|---|---|---|
| `/workspace/onnxruntime-rvnpu` | `main` | 本仓库:EP 插件、runtime、测试工具 |
| `/workspace/dayan-base` | `rvtv2/stable` | dayan 内核源码(demo/onnx)、dycc 编译器、dayan_server、设备固件 |
| `/workspace/npu-cluster-sim` | `v0.5.4-rc1` | RTL 仿真器 TestCluster |

## 2. 启动模拟器与 dayan_server

测试工具不直接连 TestCluster,而是连 dayan_server 的 socket
(`/tmp/dayan-server.sock`),由它加载固件并驱动仿真器。

手动启动(在 npu-docker 上执行):

```bash
# 1. 启动 TestCluster 仿真器
cd /workspace/npu-cluster-sim './TestCluster'
启动后输入c

# 2. 起 dayan_server（在新终端执行）
cd /workspace/dayan-base
nohup build/host/driver/dayan_server --core-mask 1 \
    > /tmp/dayan-server.log 2>&1 &
```

`./scripts/start_sim.sh` 也可以执行此脚本实现启动

注意:

- 单个 TestCluster 会话上限 **5000 万周期**,到顶即 `Verilator FAILED`
  不可恢复,重新执行 `start_sim.sh` 即归零重启;

## 3. 编译

首次准备 ONNX Runtime SDK(下载官方包解压到 `build/ort-sdk`,
下面的 cmake 参数依赖这个目录;已解压过则跳过):

```bash
./scripts/fetch_onnxruntime_sdk.sh
```

编译算子内核(编译 dayan-base 的 `demo/onnx/kernel_*.cpp`):

```bash
./scripts/build_dayan_kernels.sh
```

编译 runtime 和测试工具:

```bash
mkdir -p build/dayan-migration
cd build/dayan-migration
cmake ../.. \
  -DONNXRUNTIME_ROOT=$PWD/../../build/ort-sdk \
  -DONNXRUNTIME_LIB_DIR=$PWD/../../build/ort-sdk/lib \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```

## 4. 测试

全量算子回归(19 算子)。每个算子跑两步:先 direct 直调设备内核,
再用 ONNX Runtime EP 跑同一算子的模型对比:

```bash
export LD_LIBRARY_PATH=$PWD/build/ort-sdk/lib:$LD_LIBRARY_PATH
bash scripts/run_dayan_tests.sh                 # 全部(direct+EP 各一遍)
bash scripts/run_dayan_tests.sh <op> [<op> ...] # 只跑指定算子
bash scripts/run_dayan_tests.sh --direct-ops    # 只直调不走 EP,测周期+外推整网
```

yolo 网络真实尺寸算子测试(yolov8n_64/640 网内全部形状逐个直测,
仿真周期水位到 50M 自动重启续跑):

```bash
nohup bash scripts/run_operator_matrix.sh \
  > build/operator_matrix/nohup.log 2>&1 &

python3 tools/gen_operator_matrix_table.py        # 生成总表
# 结果:build/operator_matrix/results.csv
#       build/operator_matrix/operator_matrix_table.md
```

## 5. YOLOv8n 测试

```bash
./scripts/test_operator.sh yolov8n           # 官方 nms=True 模型(640)
./scripts/test_operator.sh yolov8n_manual    # TopK(K=1) 手动优化后处理
```

## 6. 已知限制

- TestCluster 单会话 50M 周期上限,不可续;
- conv 3×3 走标量 im2col(~80-100 周期/元素),大图在仿真中极慢;
- 大空间/部分形状在宿主高负载时仿真近乎停滞(load>50 时建议暂停跑测试);
- 设备 printf 会破坏 launch 完成,demo 内核不要用;
- 所有模型、SDK、结果和日志均生成在被忽略的 `build/` 中。
