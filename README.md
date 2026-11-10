# RVNPU ONNX Runtime Plugin EP

本项目提供独立的 `RVNPUExecutionProvider`,不修改官方 ONNX Runtime。

当前支持标准 ONNX 算子:

```text
MatMul, Add, Mul, Sub, Div, Sigmoid, Conv, MaxPool,
Concat, Split, Slice, Resize, Transpose, Softmax, ReduceMax, TopK,
ArgMax, NonMaxSuppression, GatherND
```

设备内核来自 dayan-base 仓库的 `demo/onnx/kernel_*.cpp`(含 conv 的
im2col+matmul、elementwise、softmax 等),经 dycc 编译为 `.o` 后链接进 EP。

## 1. 目录结构

```text
src/                  EP 插件、算子实现与设备 runtime
scripts/              SDK 下载、内核编译、仿真器启动、回归入口
tools/                YOLO 图片预处理与检测结果可视化
docs/                 添加算子等说明文档
tests/
  unit/               主机侧单元测试(不需要设备)
  direct/             直调设备内核的算子数值测试
  integration/        ONNX Runtime/EP 集成测试(CPU/NPU 对比)
  models/             测试模型与输入数据生成脚本
  operators/          每个算子的 prepare.sh/run.sh 测试入口
  assets/             YOLOv8n 测试图片
```

## 2. 环境要求

| 仓库 | 分支 | 用途 |
|---|---|---|
| onnxruntime(本仓库) | `main` | EP 插件、runtime、测试 |
| dayan-base | `rvtv2/stable` | dayan 内核源码(demo/onnx)、dycc 编译器、dayan_server、设备固件 |
| npu-cluster-sim | `v0.5.4-rc1` | RTL 仿真器 TestCluster |

三个仓库按同级目录并列检出即可。脚本与 CMake 默认使用同级目录
`../dayan-base`、`../npu-cluster-sim`;位置不同时,可用 `DAYAN_ROOT`、
`TC_ROOT` 环境变量或 CMake 的 `DAYAN_RUNTIME_ROOT`/`DAYAN_KERNEL_ROOT`
参数覆盖。

## 3. 首次准备与编译

首次使用前:

1. 安装 Python 依赖(YOLO 导出/图片处理需要):

```bash
pip install -r requirements.txt
```

2. 构建需要网络:首次运行会下载官方 ONNX Runtime 1.26.0 SDK(带 sha256
校验);首次跑 YOLOv8n 测试时 ultralytics 会自动下载 `yolov8n.pt` 权重,
并导出为固定参数的 ONNX 模型(版本锁定 ultralytics==8.4.92)。

一键构建(SDK 准备 + dycc 编译 dayan-base 的 `demo/onnx/kernel_*.cpp`
+ cmake + 单元测试):

```bash
./scripts/configure_build.sh
```

也可以分步执行:

```bash
./scripts/fetch_onnxruntime_sdk.sh    # 下载/校验 ONNX Runtime SDK
./scripts/build_dayan_kernels.sh      # dycc 编译设备内核

mkdir -p build/dayan-migration
cd build/dayan-migration
cmake ../.. \
  -DONNXRUNTIME_ROOT=$PWD/../../build/ort-sdk \
  -DONNXRUNTIME_LIB_DIR=$PWD/../../build/ort-sdk/lib \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```

## 4. 启动模拟器与 dayan_server

测试工具不直接连 TestCluster,而是连 dayan_server 的 socket
(`/tmp/dayan-server.sock`),由它加载固件并驱动仿真器。

手动启动:

```bash
# 1. 启动 TestCluster 仿真器
cd <npu-cluster-sim> && ./TestCluster
启动后输入 c

# 2. 起 dayan_server(在新终端执行)
cd <dayan-base>
nohup build/host/driver/dayan_server --core-mask 1 \
    > /tmp/dayan-server.log 2>&1 &
```

也可以直接执行 `./scripts/start_sim.sh` 完成上述启动。

注意:

- 单个 TestCluster 会话上限 **5000 万周期**,到顶即 `Verilator FAILED`
  不可恢复,重新执行 `start_sim.sh` 即归零重启。

## 5. 测试

全量回归(19 个算子及 YOLOv8n 真实图片测试)。每个算子跑两步:先 direct
直调设备内核,再用 ONNX Runtime EP 跑同一算子的模型对比。YOLOv8n 默认只跑
bus.jpg + manual 后处理(见第 6 节)。

```bash
bash scripts/run_dayan_tests.sh                 # 全部(direct+EP 各一遍)
bash scripts/run_dayan_tests.sh <op> [<op> ...] # 只跑指定算子
bash scripts/run_dayan_tests.sh --direct-ops    # 只直调不走 EP,测周期并外推整网
```

`--direct-ops` 的整网外推需要 `build/yolov8n_64/perf_manifest.json`
(历史测量产物);该文件不存在时只输出各样本的实测周期。

## 6. YOLOv8n 测试

```bash
bash scripts/run_dayan_tests.sh yolov8n          # bus.jpg + manual 后处理(默认)
bash scripts/run_dayan_tests.sh yolov8n-official # bus.jpg + 官方后处理
```

CPU/NPU 检测结果逐框对比,并生成画框图片:

- 预处理输入:`build/yolov8n/images/<图名>/data/`
- 检测结果与图片:`build/yolov8n/images/<图名>_<后处理>/results/`

默认测试图为 `tests/assets/bus.jpg`(另附 zidane.jpg)。换图改
`tests/operators/yolov8n/prepare.sh` 顶部的 `IMAGE` 路径,或运行时设置
`RVNPU_YOLO_IMAGE=/path/to/image.jpg`。
