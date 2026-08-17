# YOLOv8n 官方 NMS 模型算子清单

统计日期：2026-08-17

```text
Ultralytics: 8.4.92
ONNX opset: 13
输入:  images FP32 [1,3,640,640]
输出:  output0 FP32 [1,300,6]
每行:  x1、y1、x2、y2、score、class
节点数: 408
模型 SHA256: 71982116e42e6d3e280964f2d09cb8c1527c9d5016114201181767de915c5b51
```

## 算子分配

表中的分配结果来自只创建 ORT Session 的快速 assignment 检查，没有运行完整网络。

| ONNX 算子 | 总次数 | RVNPU EP | ORT/CPU |
|---|---:|---:|---:|
| Add | 9 | 8 | 1 |
| ArgMax | 1 | 0 | 1 |
| Cast | 6 | 0 | 6 |
| Concat | 24 | 17 | 7 |
| Constant | 87 | 0 | 87 |
| ConstantOfShape | 2 | 0 | 2 |
| Conv | 64 | 64 | 0 |
| Div | 2 | 1 | 1 |
| Equal | 1 | 0 | 1 |
| Expand | 2 | 0 | 2 |
| Gather | 13 | 0 | 13 |
| GatherND | 4 | 2 | 2 |
| Greater | 1 | 0 | 1 |
| MaxPool | 3 | 3 | 0 |
| Mul | 63 | 60 | 3 |
| NonMaxSuppression | 1 | 1 | 0 |
| NonZero | 4 | 0 | 4 |
| Pad | 1 | 0 | 1 |
| ReduceMax | 1 | 1 | 0 |
| Reshape | 15 | 0 | 15 |
| Resize | 2 | 2 | 0 |
| ScatterND | 1 | 0 | 1 |
| Shape | 5 | 0 | 5 |
| Sigmoid | 58 | 58 | 0 |
| Slice | 10 | 6 | 4 |
| Softmax | 1 | 1 | 0 |
| Split | 8 | 8 | 0 |
| Squeeze | 1 | 0 | 1 |
| Sub | 3 | 1 | 2 |
| Transpose | 7 | 2 | 5 |
| Unsqueeze | 7 | 0 | 7 |
| Where | 1 | 0 | 1 |
| **合计** | **408** | **235** | **173** |

官方后处理包含动态阈值筛选、索引整理、NMS 和固定 300 行输出打包。当前
`ReduceMax`、大尺寸 rank-3 `Transpose` 和 `NonMaxSuppression` 已由 RVNPU 执行；
两个读取 FP32 boxes/scores 的 `GatherND` 已由 RVNPU 执行；`ArgMax`、`Greater`、
普通 `Gather`、INT64 GatherND 及控制节点暂由 ORT/CPU 执行。
两个产生空张量的 Slice 及其 Concat 不进行有效数据计算，也由 ORT/CPU 执行。

## 快速验证

`bus.jpg` 的官方 ONNX CPU 输出与 Ultralytics Python 在相同 640×640 letterbox
输入下对比：

```text
Ultralytics detections=5
ONNX Runtime CPU detections=5
matched=5
mean IoU=0.9999995
min IoU=0.9999994
max score abs diff=3.58e-7
```

EP assignment 检查：

```text
RVNPU nodes=235
Conv=64
Concat=17
Slice=6
Transpose=2
ReduceMax=1
NonMaxSuppression=1
GatherND=2
```

当前 Plugin 对 `bus.jpg` 的完整混合推理结果：

```text
CPU detections=5
RVNPU detections=5
matched=5
mean IoU=0.988188
min IoU=0.976192
max score abs diff=0.0596533
time=10m27.735s
```

CPU、RVNPU 和叠加画框图保存在 `build/yolov8n/image_results/`。
