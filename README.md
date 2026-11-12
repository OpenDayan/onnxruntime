# Dayan ONNX Runtime Plugin EP

本项目提供独立的 `DayanExecutionProvider`,不修改官方 ONNX Runtime。

当前支持标准 ONNX 算子:

```text
MatMul, Add, Mul, Sub, Div, Sigmoid, Conv, MaxPool,
Concat, Split, Slice, Resize, Transpose, Softmax, ReduceMax, TopK,
ArgMax, NonMaxSuppression, GatherND
```

设备内核来自 dayan-base 仓库的 `demo/onnx/kernel_*.cpp`，经 dycc 编译后链接进 EP。

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
| dayan-base | `rvtv2/onnx-kernels` | 当前 ONNX 内核源码(demo/onnx)、dycc 编译器、dayan_server、设备固件 |
| npu-cluster-sim | `main` |  RTL 模拟  |
| dayan-emu | `rvtv2/stable` | C 模拟  |

配置环境变量：

```bash
export DAYAN_ROOT=/path/to/dayan-base
export RTLSIM_ROOT=/path/to/npu-cluster-sim
export DAYAN_EMU_DIR=/path/to/dayan-emu
```

## 3. 首次准备与编译

首次使用前:

1. 安装 Python 依赖:

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

mkdir -p build/dayan
cd build/dayan
cmake ../.. \
  -DDAYAN_ROOT="${DAYAN_ROOT}" \
  -DONNXRUNTIME_ROOT=$PWD/../../build/ort-sdk \
  -DONNXRUNTIME_LIB_DIR=$PWD/../../build/ort-sdk/lib \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```

## 4. 启动模拟器与 dayan_server

### RTL 模拟

```bash
bash scripts/start_rtl_sim.sh
```

`DAYAN_SERVER_SOCKET`、`DAYAN_SERVER_LOG`、`DAYAN_XDMA_DEVICE_DIR`
分别覆盖设备 socket、日志和 XDMA 目录。测试通过 `DAYAN_SERVER_SOCKET`
连接 dayan_server。

### C 模拟（dayan-emu）

先构建 dayan-emu 配套的定制 QEMU：

```bash
make -C "${DAYAN_EMU_DIR}" qemu NPU_QEMU_JOBS=8
make -C "${DAYAN_EMU_DIR}" -j4 NPU_QEMU_JOBS=8
bash scripts/start_cmodel_sim.sh
```

C 模拟启动脚本以前台运行，默认四核、`core-mask=0xf`、4096 MiB DRAM，
统一模拟器、QEMU 和 Server 的内存配置。

## 5. 测试

```bash
bash scripts/run_dayan_tests.sh                 # 全部(direct+EP 各一遍)
bash scripts/run_dayan_tests.sh <op> [<op> ...] # 只跑指定算子
```

## 6. YOLOv8n 测试

```bash
bash scripts/run_dayan_tests.sh yolov8n          # bus.jpg + manual 后处理(默认)
bash scripts/run_dayan_tests.sh yolov8n-official # bus.jpg + 官方后处理
```

CPU/NPU 检测结果逐框对比,并生成画框图片:

- 预处理输入:`build/yolov8n/images/<图名>/data/`
- 检测结果与图片:`build/yolov8n/images/<图名>_<后处理>/results/`

默认测试图为 `tests/assets/bus.jpg`(另附 zidane.jpg)。可通过
`DAYAN_YOLO_IMAGE=/path/to/image.jpg` 指定其他测试图片。
