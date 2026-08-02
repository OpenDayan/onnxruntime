# YOLOv8n ONNX 算子清单

统计日期：2026-08-02

模型配置：

```text
Ultralytics: 8.4.92
模型: YOLOv8n
ONNX opset: 13
输入: images FP32 [1,3,640,640]
输出: output0 FP32 [1,84,8400]
NMS: 未嵌入模型
模型 SHA256: 00dadd76ce2b0082e7b05f2cfd83e6d04e96ea28ff097e3dadb1c73ead6f45de
```

## 单次推理算子调用次数

模型不包含 `Loop` 或 `If`，表中将一个 ONNX 节点记作一次算子调用。`Constant` 可能在 ORT 创建 Session 时处理。

| ONNX 算子 | 总次数 | RVNPU EP | 未被 RVNPU 接管 |
|---|---:|---:|---:|
| Add | 9 | 8 | 1 |
| Concat | 17 | 17 | 0 |
| Constant | 24 | 0 | 24 |
| Conv | 64 | 64 | 0 |
| Div | 2 | 1 | 1 |
| Gather | 1 | 0 | 1 |
| MaxPool | 3 | 3 | 0 |
| Mul | 60 | 58 | 2 |
| Reshape | 8 | 0 | 8 |
| Resize | 2 | 2 | 0 |
| Shape | 1 | 0 | 1 |
| Sigmoid | 58 | 58 | 0 |
| Slice | 2 | 2 | 0 |
| Softmax | 1 | 1 | 0 |
| Split | 8 | 8 | 0 |
| Sub | 2 | 2 | 0 |
| Transpose | 1 | 1 | 0 |
| **合计** | **263** | **225** | **38** |

FP32 计算和数据整理节点均已有 RVNPU EP 接管实现。`Constant`、`Shape`、`Gather`、`Reshape` 以及 1 个 INT64 Add、1 个 INT64 Div、2 个 INT64 Mul 由 ORT 处理；这些节点只生成或计算形状元数据。

`TopK` 和 `NonMaxSuppression` 不在这份模型中，因为导出时使用了 `nms=False`；它们目前通过独立 ONNX 子图测试。

## 本轮实际运行结果

为缩短调试时间，本轮分别选择新算子在真实 YOLOv8n 图中执行，其他节点由 CPU EP 完成。数据整理算子、逐元素算子和 Softmax 均通过实际 assignment 检查。

```text
Concat/Split/Slice/Resize/Transpose: max_abs_diff=0
Add/Mul/Div（仅FP32节点）:           max_abs_diff=0
Sub:                                max_abs_diff=0
Softmax:                            max_abs_diff=0.00012207
Softmax max_error_percent:          0.0000191665%
```

移除 INT64 device kernel 后，在真实 YOLOv8n 中只启用 Add、Mul、Div 进行检查：

```text
Add=8, Mul=58, Div=1
CPU vs RVNPU max_abs_diff=0
```

完整 225 个 NPU 节点的整网严格数值测试本轮没有重复执行。此前包含 BF16 Conv 的整网结果最大相对误差约为 7.1%；完整网络精度仍主要受现有 BF16 Conv 路径影响。

调试某一种算子时可使用：

```bash
RVNPU_YOLO_OPERATOR=Softmax ./scripts/test_operator.sh yolov8n
```
