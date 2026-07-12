# RVNPU ONNX Runtime Plugin EP

本项目为 ONNX Runtime 提供独立的 `RVNPUExecutionProvider`。

当前已经跑通：

```text
tiny CNN
  Conv -> Relu -> MaxPool -> Flatten -> MatMul
   CPU     CPU       CPU        CPU        RVNPU
```

当前 NPU 支持范围是 opset 13、FP32、静态二维输入的 `ai.onnx::MatMul`。

## 1. 准备开发环境

项目在 RVNPU 开发容器中构建和测试。默认目录如下：

```text
/workspace/onnxruntime-rvnpu   本项目
/workspace/RVNPU-compiler      RVNPU 编译器
/workspace/npu-kernels         device kernel 和 ABI bridge
/workspace/npu-runtime         host runtime
/workspace/npu-qemu            QEMU
/workspace/NPU-Setup           emulator 启动环境
/workspace/llvm-build          RVNPU LLVM 构建目录
```

### 准备 NPU-Setup

在 Linux 开发容器中编译 emulator、QEMU 和 firmware：

```bash
cd /workspace/NPU-Setup
./3.compile_emu.sh
./4.compile_qemu.sh
./5.compile_firmware.sh
```

如果是标准的全新 `NPU-Setup` 仓库，并且 `workspace/` 中还没有各依赖仓库，需要先执行一次：

```bash
./1.init_repo.sh
```

当前开发容器已经把各仓库放在 `/workspace/`，不需要重复执行 `1.init_repo.sh`。本项目也不需要执行 `2.compile_kernels.sh`，MatMul device ELF 会由 `run_e2e.sh` 单独构建。测试时不需要手动运行 `6.0.run_emu.sh`，测试脚本会自动启动和关闭 emulator。


## 2. 第一次运行前安装依赖

进入项目目录：

```bash
cd /workspace/onnxruntime-rvnpu
```

下载固定版本的官方 ONNX Runtime C/C++ SDK：

```bash
./scripts/fetch_onnxruntime_sdk.sh
```

安装生成 ONNX 模型和 reference 数据所需的 Python 包：

```bash
python3 -m pip install --user -r requirements.txt
```

## 3. 设置依赖路径

在当前容器终端执行：

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

## 4. 验证环境和完整执行链

`run_e2e.sh` 用于第一次配置环境后的整体检查，以及修改公共框架后的完整回归：

```bash
./scripts/run_e2e.sh
```

脚本会自动完成：

1. 构建 `librvnpu_ep.so` 和测试工具；
2. 通过 RVNPU-compiler 构建 MatMul device ELF；
3. 生成测试数据并运行 MatMul 测试。

正常输出如下：

```text
[1/3] 构建 Plugin ... PASS
[2/3] 构建 device ELF ... PASS
[3/3] 验证 MatMul ... PASS

全部验证通过。
```

成功后主要产物是：

```text
build/linux/librvnpu_ep.so
build/device/device_build/user_triton_matmul.elf
```

## 5. 查看失败原因

脚本使用临时日志。成功后自动删除；失败时在终端显示最后 80 行。

## 6. 单算子测试

日常开发或新增算子时，不需要反复运行完整 E2E。测试 MatMul 使用：

```bash
./scripts/test_operator.sh matmul
```

预计输出：

```text
[1/2] Direct kernel 数值测试
      M=1,N=32,K=32       PASS max_abs_diff=0
      M=32,N=32,K=32      PASS max_abs_diff=0
      M=3,N=5,K=7         PASS max_abs_diff=0

[2/2] tiny CNN：CPU/reference/RVNPU 输出对比
      CPU vs ReferenceEvaluator max_abs_diff=0
      RVNPU vs golden max_abs_diff=0
```
