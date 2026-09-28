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
| `/workspace/npu-cluster-sim` | `main`(测试实验版本为) | RTL 仿真器 TestCluster |

## 2. 启动模拟器与 dayan_server

测试工具不直接连 TestCluster,而是连 dayan_server 的 socket
(`/tmp/dayan-server.sock`),由它加载固件并驱动仿真器:

```bash
./scripts/start_sim.sh
```

该脚本会清理旧实例、在 tmux 会话 `sim` 里启动 TestCluster 并开始仿真、
拉起 dayan_server(core-mask=1 等必要参数已内置)。手动查看周期水位:

```bash
tmux capture-pane -t sim -p | grep -oE "Cycl[e]*[0-9]+" | tail -1
```

注意:

- 单个 TestCluster 会话上限 **5000 万周期**,到顶即 `Verilator FAILED`
  不可恢复,重新执行 `start_sim.sh` 即归零重启;

## 3. 编译

内核(改了 `demo/onnx/kernel_*.cpp` 或 dayan_intrinsics.h 后执行):

```bash
./scripts/build_dayan_kernels.sh
```

runtime 与测试工具(增量):

```bash
cd build/dayan-migration && cmake . && make -j8
```

direct 测试需要 dummy ELF 参数(值无用):

```bash
export RVNPU_KERNEL_ELF=/tmp/dummy-kernel-elf && : > /tmp/dummy-kernel-elf
```

## 4. 测试

全量算子回归(direct + EP,19 算子):

```bash
export LD_LIBRARY_PATH=$PWD/build/ort-sdk/lib:$LD_LIBRARY_PATH
bash scripts/run_dayan_tests.sh                 # 除 yolov8n* 外全部
bash scripts/run_dayan_tests.sh <op> [<op> ...] # 指定算子
bash scripts/run_dayan_tests.sh --direct-ops      # 小样本周期测量 + 整网外推
```

算子矩阵(两网全部形状逐个直测,自动管理水位/重启):

```bash
nohup bash scripts/run_operator_matrix.sh \
  > build/operator_matrix/nohup.log 2>&1 &
tail -3 build/operator_matrix/run.log             # 进度
python3 tools/gen_operator_matrix_table.py        # 生成总表
# 结果:build/operator_matrix/results.csv
#       build/operator_matrix/operator_matrix_table.md
```

单算子旧入口仍可用:

```bash
./scripts/test_operator.sh <op>
```

`RVNPU_CONV_REFERENCE=bf16` 可将 direct Conv 的 CPU reference 切为 BF16。

周期测量注意:超时按 `实测周期 ÷ 仿真速度(~4-6k 周期/秒) × 1.5` 估,
不拍脑袋;宿主机 load 高时仿真速度骤降,先 `uptime` 排除。

## 5. YOLOv8n 测试

```bash
./scripts/test_operator.sh yolov8n           # 官方 nms=True 模型(640)
./scripts/test_operator.sh yolov8n_manual    # TopK(K=1) 手动优化后处理
```

整网未在仿真中跑通(周期需求 >100M,超单会话上限);64 输入缩小模型
`build/yolov8n_64/yolov8n_64.onnx` 用于周期外推。

## 6. 已知限制

- TestCluster 单会话 50M 周期上限,不可续;
- conv 3×3 走标量 im2col(~80-100 周期/元素),大图在仿真中极慢;
- 大空间/部分形状在宿主高负载时仿真近乎停滞(load>50 时建议暂停跑测试);
- 设备 printf 会破坏 launch 完成,demo 内核不要用;
- 所有模型、SDK、结果和日志均生成在被忽略的 `build/` 中。
