# RVNPU ONNX Runtime Plugin EP

本项目提供独立的 `RVNPUExecutionProvider`，不修改官方 ONNX Runtime。

当前支持标准 ONNX 算子：

```text
MatMul, Add, Mul, Sub, Div, Sigmoid, Conv, MaxPool,
Concat, Split, Slice, Resize, Transpose, Softmax, ReduceMax, TopK,
ArgMax, NonMaxSuppression, GatherND
```

YOLOv8n 默认使用 Ultralytics 官方 `nms=True` ONNX，输出 FP32
`[1,300,6]`，每行是 `x1、y1、x2、y2、score、class`。主网络中已支持的
FP32 节点由 RVNPU 执行；官方后处理中的 ReduceMax、ArgMax、INT64 索引
Transpose/Slice、NMS、两个 FP32 GatherND 和一个 INT64 GatherND 已接管。
Greater、NonZero、三个静态零宽分支、普通 Gather 和小型控制节点由 ORT/CPU 执行。

## 1. 准备开发环境

项目在 RVNPU Linux 开发容器中构建。当前开发分支和默认目录如下：

```text
/workspace/onnxruntime-rvnpu                     main
/workspace/npu-setup                             rvtv2/stable
/workspace/npu-setup/workspace/npu-kernels       rvtv2/onnx
/workspace/npu-setup/workspace/npu-runtime       rvtv2/stable
/workspace/npu-setup/workspace/npu-firmware      rvtv2/stable
/workspace/npu-setup/workspace/npu-qemu          rvtv2/stable
/workspace/npu-setup/workspace/rvtng-cmodel      rvtv2/stable
/workspace/triton-rvnpu                          rvtv2/stable
/workspace/llvm-project/install                  Triton lowering 使用的 LLVM
/opt/llvm                                        npu-kernels 编译使用的 LLVM
```

全新环境先在 `NPU-Setup` 拉取依赖：

```bash
cd /workspace/npu-setup
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
export NPU_SETUP_ROOT=/workspace/npu-setup
export NPU_WORKSPACE_DIR="$NPU_SETUP_ROOT/workspace"
export NPU_KERNELS_ROOT="$NPU_WORKSPACE_DIR/npu-kernels"
export NPU_RUNTIME_ROOT="$NPU_WORKSPACE_DIR/npu-runtime"
export NPU_QEMU_ROOT="$NPU_WORKSPACE_DIR/npu-qemu"
export RVNPU_COMPILER_ROOT=/workspace/triton-rvnpu/third_party/riscvnpu
export RVNPU_COMPILER_REPO_ROOT=/workspace/triton-rvnpu
export TRITON_BUILD_DIR=/workspace/triton-rvnpu/build
export LLVM_BUILD_DIR=/workspace/llvm-project/install
export DEVICE_LLVM_BUILD_DIR=/opt/llvm
export PYTHON_EXECUTABLE=python3

说明：`LLVM_BUILD_DIR` 用于 Triton lowering；`DEVICE_LLVM_BUILD_DIR`
用于编译/链接 `npu-kernels/rvtv2/onnx` 中的 C/C++ device kernels。
当前两套 LLVM 的 RVT headers/内建函数版本不同，不能混用。
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
./scripts/test_operator.sh silu
./scripts/test_operator.sh conv
# 使用 CPU BF16 reference 对照 Conv（默认仍为 FP32 reference）
RVNPU_CONV_REFERENCE=bf16 ./scripts/test_operator.sh conv
./scripts/test_operator.sh maxpool
./scripts/test_operator.sh concat
./scripts/test_operator.sh split
./scripts/test_operator.sh slice
./scripts/test_operator.sh resize
./scripts/test_operator.sh transpose
./scripts/test_operator.sh softmax
./scripts/test_operator.sh reduce_max
./scripts/test_operator.sh topk
./scripts/test_operator.sh nms
./scripts/test_operator.sh selection
```

`RVNPU_CONV_REFERENCE` 只切换 direct Conv 数值测试中的 CPU reference，不会修改
ONNX Runtime 的 CPU EP；默认值为 `fp32`。`bf16` 模式保留 FP32 输出和累加，
仅将输入、权重按已生成 Triton pack 阶段的规则取 FP32 高 16 位，转换为 BF16。

每项测试包含 device kernel 数值测试，以及一次 ORT CPU/RVNPU 输出对比。程序会在内部确认目标节点确实分配给 RVNPU。`selection` 覆盖 ArgMax、INT64 GatherND、INT64 Transpose/Slice 和 Greater/NonZero 留 CPU 的空候选链路；NMS 测试还包含 `Split → Transpose → NMS → GatherND` 子图。

## 6. YOLOv8n 测试

首次导出模型前安装固定版本：

```bash
python3 -m pip install --user ultralytics==8.4.92
```

执行：

```bash
./scripts/test_operator.sh yolov8n
```

模型为 opset 13、FP32、静态 `[1,3,640,640]`。准备阶段生成官方
`nms=True` 模型 `build/yolov8n/yolov8n.onnx`，该命令默认运行主网络和官方
后处理，输出 `[1,300,6]`。官方后处理中的未支持节点自动回退到 ORT/CPU。

NMS 的受限算子契约和小后处理子图继续由 `./scripts/test_operator.sh nms` 覆盖。

指定真实图片测试并画框：

```bash
./scripts/test_operator.sh yolov8n \
  /home/dev/.local/lib/python3.10/site-packages/ultralytics/assets/bus.jpg
```

结果保存在 `build/yolov8n/image_results/`。

手动优化版使用 `TopK(K=1)` 一次取得最高类别分数和类别编号，替代官方后处理中的
`ReduceMax + ArgMax`，输出仍为 `[1,300,6]`：

```bash
./scripts/test_operator.sh yolov8n_manual
./scripts/test_operator.sh yolov8n_manual \
  /home/dev/.local/lib/python3.10/site-packages/ultralytics/assets/bus.jpg
```

手动模型保存在 `build/yolov8n/yolov8n_manual.onnx`，图片结果保存在
`build/yolov8n/manual_image_results/`。官方 `yolov8n.onnx` 不做修改。

所有模型、SDK、ELF 和日志均生成在被忽略的 `build/` 中。
