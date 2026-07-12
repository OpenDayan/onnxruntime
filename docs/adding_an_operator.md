# CNN 算子列表与新增算子指南

## 1. 算子需求列表


```text
Input
  -> Conv
  -> Relu
  -> MaxPool
  -> Flatten
  -> MatMul
  -> Output
```


## 2. 一个 ONNX 算子怎样到达 NPU

当前 MatMul 的完整调用链如下：

```text
ONNX Runtime 读取 ai.onnx::MatMul
  -> RvnpuEp::SupportsMatMul
       文件：src/plugin/ep.cpp
       作用：判断这个节点是否属于 RVNPU 当前支持范围

  -> RvnpuEp::GetCapabilityImpl
       文件：src/plugin/ep.cpp
       作用：向 ORT 声明该 MatMul 可以交给 RVNPU

  -> CreateKernelRegistry
       文件：src/plugin/kernel_registry.cpp
       作用：让 ORT 找到 MatMul 对应的 RVNPU kernel 实现

  -> MatMulKernel::ComputeImpl
       文件：src/ops/matmul/matmul.cpp
       作用：读取 ONNX tensor、检查形状、创建输出

  -> DeviceContext::RunMatMul
       文件：src/runtime/device_context.cpp
       作用：申请 device buffer、拷贝输入、组织 kernel 参数

  -> npu_runtime_kernel_start
       来源：npu-runtime
       作用：启动 device ELF 中的稳定导出符号

  -> triton_matmul_fp32_kernel
       文件：$NPU_KERNELS_ROOT/kernel/triton_matmul_bridge.cpp
       作用：构造编译器内部 ABI、计算 Triton grid

  -> matmul_kernel
       来源：Triton Python 经 RVNPU-compiler 生成
       作用：执行真正的 MatMul 计算

  -> QEMU + cmodel
```


## 3. 以新增 Relu 为例，要改哪些位置

| 调用链位置 | MatMul 当前实现 | 新增 Relu 时对应工作 |
|---|---|---|
| ONNX 能力判断 | `SupportsMatMul` | 增加 `SupportsRelu`，检查 opset、FP32、shape |
| ORT kernel | `MatMulKernel` | 新建 `ReluKernel` |
| ONNX tensor 解析 | `MatMulKernel::ComputeImpl` | 实现 `ReluKernel::ComputeImpl` |
| runtime 调用 | `DeviceContext::RunMatMul` | 增加 `DeviceContext::RunRelu` |
| kernel 名 | `kMatMulFp32` | 增加例如 `kReluFp32` |
| device 稳定入口 | `triton_matmul_fp32_kernel` | 提供例如 `triton_relu_fp32_kernel` |
| 编译器内部函数 | `matmul_kernel` | 由 RVNPU-compiler 生成 `relu_kernel` |
| 测试目录 | `tests/operators/matmul/` | 新建 `tests/operators/relu/` |

当前结构下，每个新算子通常需要一个清楚的 `DeviceContext::Run<Operator>`。它负责该算子独有的 buffer 数量、参数顺序和 kernel 启动。runtime 初始化、ELF 加载、错误处理和 `DeviceBuffer` 仍然复用公共实现。

## 4. 新增一个算子的实际步骤

### 第一步：定义支持范围

先写清楚第一版究竟支持什么。例如 Relu 可以定义为：

```text
domain: ai.onnx
op: Relu
opset: 13
dtype: FP32
shape: 静态连续 tensor
```

如果 device kernel 暂时不支持某种 dtype、动态 shape、广播或属性，`GetCapability` 就不能声明支持。合法但不支持的节点应留给 CPU。

### 第二步：准备 device kernel

如果使用 Triton：

```text
Triton Python Relu
  -> RVNPU-compiler
  -> 编译器内部 relu_kernel
  -> npu-kernels 中的稳定 ABI bridge
  -> 包含 triton_relu_fp32_kernel 的 device ELF
```

这一阶段需要完成：

1. 确认 Triton 算子能被 RVNPU-compiler lowering；
2. 明确内部函数 ABI 和 grid；
3. 在 `npu-kernels` 提供稳定导出入口；
4. 把内部函数和 bridge 链接进测试 device ELF；
5. 先用不经过 ORT 的 direct probe 与 CPU reference 对比。

只有 direct probe 正确，才进入 Plugin EP 接入。

### 第三步：登记 kernel 名并实现 runtime 调用

修改：

```text
src/runtime/kernel_catalog.h
src/runtime/device_context.h
src/runtime/device_context.cpp
```

需要完成：

1. 在 `kernel_catalog.h` 集中记录 device 导出符号；
2. 加载 ELF 后通过 `RequireKernel` 找到对应 descriptor；
3. 增加 `Run<Operator>`；
4. 在一个设备锁内完成 buffer 分配、H2D、kernel launch、D2H；
5. 所有标量按 device ABI 明确转换宽度。

不要在 `src/ops/` 中直接重复 `npu_runtime_init`、内存管理或 ELF 加载。

### 第四步：实现 ONNX Runtime kernel

新建：

```text
src/ops/<算子名>/<算子名>.h
src/ops/<算子名>/<算子名>.cpp
```

`ComputeImpl` 只负责：

1. 读取 ONNX 输入和属性；
2. 再次检查运行时 dtype 和 shape；
3. 创建正确形状的 ORT 输出；
4. 调用 `DeviceContext::Run<Operator>`；
5. 将异常转换为带上下文的 `OrtStatus`。

参考文件是：

```text
src/ops/matmul/matmul.h
src/ops/matmul/matmul.cpp
```

### 第五步：注册算子并声明能力

修改：

```text
src/plugin/kernel_registry.cpp
src/plugin/ep.h
src/plugin/ep.cpp
CMakeLists.txt
```

分别完成：

1. 用 `RVNPU_DEFINE_KERNEL` 定义 ONNX domain、opset 和 type constraint；
2. 把新 kernel builder 加入 `kBuilders`；
3. 增加 `Supports<Operator>`；
4. 在 `GetCapabilityImpl` 中只接管通过检查的节点；
5. 将新 `.cpp` 加入 `rvnpu_ep` 构建目标。

能力判断和 `ComputeImpl` 的检查必须一致。不能在 capability 阶段宣称支持全部 Relu，运行时再发现 kernel 实际无法执行。

### 第六步：添加独立测试套件

新建：

```text
tests/operators/<算子名>/prepare.sh
tests/operators/<算子名>/run.sh
```

`prepare.sh` 生成：

```text
标准 ONNX 模型
确定性输入
CPU / ONNX ReferenceEvaluator reference
```

`run.sh` 验证：

1. direct probe 数值正确；
2. CPU、reference 与 RVNPU 输出一致。

`test_operator.sh` 根据算子名直接执行这两个文件。

## 5. 如何运行新增算子的测试

先只测试目标算子：

```bash
./scripts/test_operator.sh relu
```

通过后运行完整流程：

```bash
./scripts/run_e2e.sh
```

## 6. 完成一个算子的判断标准

一个算子只有同时满足以下条件才算接入完成：

- device direct probe 正确；
- ORT assignment 明确显示目标节点由 RVNPU 执行；
- 混合模型结果与 CPU/reference 对齐；
- 连续执行稳定；
- 完整 E2E 通过。
