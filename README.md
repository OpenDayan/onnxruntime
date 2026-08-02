# RVNPU ONNX Runtime Plugin EP

本项目提供独立的 `RVNPUExecutionProvider`，不修改官方 ONNX Runtime。

当前支持标准 ONNX 算子：

```text
MatMul, Add, Mul, Sub, Div, Sigmoid, Conv, MaxPool,
Concat, Split, Slice, Resize, Transpose, Softmax, TopK, NonMaxSuppression
```

当前 YOLOv8n 的 FP32 计算节点和数据整理节点均具备 RVNPU 接管实现。`Constant`、`Shape`、`Gather`、`Reshape` 以及 4 个 INT64 形状计算节点留给 ORT 处理。

## 1. 准备开发环境

项目在 RVNPU Linux 开发容器中构建。当前开发分支和默认目录如下：

```text
/workspace/onnxruntime-rvnpu   rvtv2/yolov8n-ops
/workspace/RVNPU-compiler      rvtv2/dev
/workspace/npu-kernels         rvtv2/onnx-kernel-opt
/workspace/npu-runtime         rvtv2/stable
/workspace/npu-firmware        rvtv2/stable
/workspace/npu-qemu            rvtv2/stable
/workspace/rvtng-cmodel        rvtv2/stable
/workspace/NPU-Setup           rvtv2/stable
/workspace/llvm-build          RVNPU LLVM 构建目录
```

全新环境先在 `NPU-Setup` 拉取依赖：

```bash
cd /workspace/NPU-Setup
./1.init_repo.sh
```

仓库已经位于 `/workspace` 时，不要重复执行。构建 emulator、QEMU 和 firmware：

```bash
./3.compile_emu.sh
./4.compile_qemu.sh
./5.compile_firmware.sh
```

测试脚本会自动启动和关闭 emulator，不需要手动运行 `6.0.run_emu.sh`。

## 2. 安装依赖

```bash
cd /workspace/onnxruntime-rvnpu
./scripts/fetch_onnxruntime_sdk.sh
python3 -m pip install --user -r requirements.txt
```

脚本下载固定版本的 ONNX Runtime 1.26.0 C/C++ SDK，不需要另行 clone 官方 `onnxruntime` 仓库。

## 3. 设置路径

```bash
export ONNXRUNTIME_ROOT=/workspace/onnxruntime-rvnpu/build/ort-sdk
export ONNXRUNTIME_LIB_DIR="$ONNXRUNTIME_ROOT/lib"
export RVNPU_COMPILER_ROOT=/workspace/RVNPU-compiler
export NPU_KERNELS_ROOT=/workspace/npu-kernels
export NPU_RUNTIME_ROOT=/workspace/npu-runtime
export NPU_QEMU_ROOT=/workspace/npu-qemu
export NPU_SETUP_ROOT=/workspace/NPU-Setup
export LLVM_BUILD_DIR=/workspace/llvm-build
export PYTHON_EXECUTABLE=python3
```

## 4. 首次构建

```bash
./scripts/run_e2e.sh
```

它会构建 Plugin、生成包含全部 kernel 的 device ELF，并运行 MatMul/tiny CNN 回归。主要产物：

```text
build/linux/librvnpu_ep.so
build/device/device_build/rvnpu_kernels.elf
```

## 5. 单算子测试

```bash
./scripts/test_operator.sh matmul
./scripts/test_operator.sh add
./scripts/test_operator.sh mul
./scripts/test_operator.sh sub
./scripts/test_operator.sh div
./scripts/test_operator.sh sigmoid
./scripts/test_operator.sh conv
./scripts/test_operator.sh maxpool
./scripts/test_operator.sh concat
./scripts/test_operator.sh split
./scripts/test_operator.sh slice
./scripts/test_operator.sh resize
./scripts/test_operator.sh transpose
./scripts/test_operator.sh softmax
./scripts/test_operator.sh topk
./scripts/test_operator.sh nms
```

每项测试包含 device kernel 数值测试，以及一次 ORT CPU/RVNPU 输出对比。程序会在内部确认目标节点确实分配给 RVNPU。NMS 测试还包含标准 `TopK → NMS` 后处理子图。

## 6. YOLOv8n 测试

首次导出模型前安装固定版本：

```bash
python3 -m pip install --user ultralytics==8.4.92
```

执行：

```bash
./scripts/test_operator.sh yolov8n
```

模型为 opset 13、FP32、静态 `[1,3,640,640]`，不包含 NMS。脚本会在内部检查实际 EP assignment，并分别运行一次 CPU 和 RVNPU。

只验证 YOLO 中某一个 RVNPU 算子时：

```bash
RVNPU_YOLO_OPERATOR=Softmax ./scripts/test_operator.sh yolov8n
```

所有模型、SDK、ELF 和日志均生成在被忽略的 `build/` 中。
