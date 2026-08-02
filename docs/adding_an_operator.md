# 新增算子支持

## 1. 当前支持范围

| ONNX 算子 | 首版范围 | device 入口 |
|---|---|---|
| MatMul | FP32 输入输出、静态 rank-2、单次 BF16 计算 | `triton_matmul_fp32_kernel` |
| Add / Mul | FP32 rank 0～4 广播 | `kernel_add_fp32` / `kernel_mul_fp32` |
| Sub / Div | FP32 rank 0～4 | `kernel_sub_fp32` / `kernel_div_fp32` |
| Sigmoid | FP32、静态 rank 1～4 | `sigmoid_f32` |
| Conv | FP32 输入输出、单次 BF16 MatMul、batch=1、group=1、1×1/3×3、stride 1/2、对称 padding | `im2col_nchw_fp32` + MatMul + Add |
| MaxPool | FP32 NCHW、2×2/3×3/5×5、stride 1/2、对称 padding | `maxpool2d_nchw_fp32` |
| Concat / Split / Slice | FP32 rank 1～4；Slice 只支持单 axis、step=1 | `concat_fp32` / `split_fp32` / `slice_fp32` |
| Resize | FP32 NCHW、nearest、asymmetric、floor、整数倍放大 | `resize_nearest_nchw_fp32` |
| Transpose | FP32 rank-4、perm=[0,2,1,3] | `transpose_0213_fp32` |
| Softmax | FP32 rank-4、axis=1、axis 维为 16 | `softmax_axis1_fp32` |
| TopK | FP32 rank-2、axis=-1、largest/sorted、常量 K≤300 | `topk_lastdim_fp32` |
| NonMaxSuppression | corner boxes、batch=1、常量 max output≤300 | `non_max_suppression_fp32` |

YOLOv8n 中的 `Constant`、`Shape`、`Gather`、`Reshape` 和单元素 INT64 Add/Div/Mul 留给 ORT 处理；它们生成或计算形状元数据，不需要 device kernel。

`origin/conv-kernel` 已有 BF16/低比特直接 Conv 导出，但其 host 测试仍使用旧版 npu-runtime 加载 ABI；当前仅作 kernel 设计参考，尚未接入 Plugin EP。

## 2. 调用链

以 Conv 为例：

```text
ONNX ai.onnx::Conv
  -> RvnpuEp::SupportsConv
       src/plugin/ep.cpp
       检查 schema 版本、dtype、静态形状和属性

  -> ConvKernel::ComputeImpl
       src/ops/conv/conv.cpp
       读取 ORT tensor，创建输出

  -> DeviceContext::RunConv
       src/runtime/device_context.cpp
       分配 device buffer，完成 H2D、launch、D2H

  -> im2col_nchw_fp32
       npu-kernels/kernel/kernel_im2col.cpp

  -> triton_matmul_fp32_kernel
       npu-kernels/kernel/triton_matmul_bridge.cpp

  -> matmul_kernel
       RVNPU-compiler 从 integration/kernels/matmul.py 生成

  -> QEMU + cmodel
```

MatMul 和 Conv 的接口仍接收、输出 FP32 tensor，内部使用一次 BF16 MatMul 和 FP32 累加。

## 3. 新增一个算子要改哪些文件

以现有 MatMul 为参考：

```text
src/ops/matmul/matmul.h/.cpp
  ONNX 属性、输入输出和 ComputeImpl

src/plugin/ep.h/.cpp
  SupportsMatMul 和 capability 声明

src/plugin/kernel_registry.cpp
  注册标准 ONNX kernel、schema version 和类型约束

src/runtime/kernel_catalog.h
  记录 device ELF 导出符号

src/runtime/device_context.h/.cpp
  RunMatMul、buffer、参数顺序和 kernel launch

npu-kernels/kernel/triton_matmul_bridge.cpp
  npu-runtime 可调用的稳定 device ABI

tests/operators/matmul/prepare.sh
tests/operators/matmul/run.sh
tools/test_rvnpu_matmul.cpp
  device kernel 数值测试

CMakeLists.txt
  加入新源文件和测试工具
```

如果算子不经过 Triton，可参考 `npu-kernels/kernel/kernel_topk.cpp` 或
`npu-kernels/kernel/kernel_nms.cpp`，Plugin 侧结构不变。

## 4. 添加步骤

1. 查 ONNX schema，写明 opset、dtype、shape、属性、广播和输出语义。
2. 在 `npu-kernels` 实现稳定 device 入口，先用 device kernel 测试对比 CPU。
3. 在 `kernel_catalog.h` 登记符号，在 `DeviceContext` 增加 `Run<Operator>`。
4. 新建 `src/ops/<name>/`，实现 ORT kernel。
5. 在 `ep.cpp` 只接管确定支持的节点；其他合法配置留给 CPU。
6. 在 `kernel_registry.cpp` 和 `CMakeLists.txt` 注册。
7. 新建 `tests/operators/<name>/prepare.sh` 与 `run.sh`。
8. 运行：

```bash
./scripts/build_device_kernel.sh
./scripts/configure_build.sh
./scripts/test_operator.sh <name>
```

测试包含 device kernel 数值检查，以及标准 ONNX 模型的一次 CPU/RVNPU 输出对比。ORT 测试必须在内部确认目标节点确实分配给 RVNPU。
