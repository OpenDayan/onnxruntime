# YOLOv5 与 YOLOv8 算子需求对照

以下以 FP32、batch=1、640×640、静态 ONNX 检测模型为基准。当前实际验证模型为
Ultralytics 8.4.92 导出的 YOLOv8n、opset 13，输出 `[1,300,6]`，包含官方 NMS。
YOLOv5 的算子会随导出版本和是否包含后处理而变化。

| 功能分类 | ONNX 算子 | 简单功能 | YOLOv5 | 当前 YOLOv8n | 当前本地分支 |
|---|---|---|---|---|---|
| 主干计算 | `Conv` | 提取图像特征，包括普通卷积和 1×1 卷积 | 需要 | 需要 | **已支持** |
| 主干计算 | `Sigmoid` | 将数值压缩到 0～1，也是 SiLU 的一部分 | 需要 | 58 个 | **58 个分配；其中 57 个参与 SiLU 融合** |
| 主干计算 | `Mul` | 逐元素乘法，和 Sigmoid 组成 SiLU | 需要 | 61 个 | **60 个分配；其中 57 个参与 SiLU 融合** |
| 主干计算 | `Add` | 残差连接和逐元素加法 | 需要 | 9 个 | **8 个由 Dayan 执行** |
| 主干计算 | `MaxPool` | 聚合局部特征，SPPF 模块使用 | 需要 | 需要 | **已支持** |
| 多尺度融合 | `Resize` | 放大特征图，使不同尺度能够融合 | 需要 | 2 个 | **已支持** |
| 多尺度融合 | `Concat` | 拼接不同层或不同尺度的特征图 | 需要 | 24 个 | **17 个由 Dayan 执行** |
| 数据整理 | `Constant` | 提供模型中预先保存的形状、索引和网格数据 | 视导出图而定 | 87 个 | **ORT 初始化处理** |
| 数据整理 | `Reshape` | 改变张量形状，不改变元素值 | 需要 | 15 个 | **ORT 元数据处理** |
| 数据整理 | `Transpose` | 调换张量维度顺序 | 视导出图而定 | 7 个 | **7 个由 Dayan 执行** |
| 数据整理 | `Split` | 将一个张量拆成多份 | 视导出图而定 | 8 个 | **已支持** |
| 数据整理 | `Slice` | 从一个张量中取出连续区域 | 视导出图而定 | 10 个 | **8 个由 Dayan 执行，2 个零宽分支留 CPU** |
| 数据整理 | `Gather` | 根据索引选择数据 | 视导出图而定 | 13 个 | **ORT 处理** |
| 数据整理 | `Shape` | 取得张量各维度的大小 | 动态图常见 | 5 个 | **ORT 处理** |
| 数据整理 | `Unsqueeze` | 增加一个长度为 1 的维度 | 视导出图而定 | 7 个 | **ORT 处理** |
| 数据整理 | `Squeeze` | 删除长度为 1 的维度 | 视导出图而定 | 1 个 | **ORT 处理** |
| YOLOv5 框解码 | `Pow` | 某些导出图用平方计算 anchor 调整后的框宽高 | 视导出图而定 | 不包含 | **未实现** |
| YOLOv8 DFL | `Softmax` | 将每条边的 16 个距离分数转换成概率 | 不需要 | 1 个 | **已支持** |
| YOLOv8 DFL | `MatMul` | 某些导出图用固定权重对距离概率加权求和 | 不需要 | 当前模型不包含 | **已支持受限节点** |
| 边框解码 | `Sub` | 用候选点坐标减去左边、上边距离 | 视导出图而定 | 3 个 | **1 个由 Dayan 执行** |
| 边框解码 | `Div` | 坐标或比例换算 | 视导出图而定 | 2 个 | **1 个由 Dayan 执行** |
| 类别选择 | `ReduceMax` | 取得每个候选框的最高类别分数 | 可选后处理 | 官方 1 个，手动 0 个 | **已支持** |
| 类别选择 | `ArgMax` | 取得最高分对应的类别编号 | 可选后处理 | 官方 1 个，手动 0 个 | **已支持** |
| 候选筛选 | `Greater` | 将最高类别分数与置信度阈值比较 | 可选后处理 | 1 个 | **CPU EP 处理** |
| 候选筛选 | `NonZero` | 将 BOOL mask 压缩为有效候选索引 | 可选后处理 | 4 个 | **CPU EP 处理** |
| 类别选择 | `OneHot` | 将类别编号变成类别选择向量 | 可选后处理 | 不包含 | **未实现** |
| 类别选择 | `TopK` | 手动版用 K=1 同时取得最高分和类别编号 | 可选后处理 | 官方 0 个，手动 1 个 | **已支持** |
| 检测后处理 | `NonMaxSuppression` | 删除指向同一物体的重叠候选框 | 可选后处理 | 1 个 | **已支持** |
| 检测后处理 | `GatherND` | 按索引取出候选 boxes、scores 和类别编号 | 可选后处理 | 4 个 | **2 个 FP32 和 1 个 INT64 节点由 Dayan 执行** |

当前 `rvtv2/yolov8n-ops` 已注册：

```text
MatMul, Add, Mul, Sub, Div, Sigmoid, Conv, MaxPool,
Concat, Split, Slice, Resize, Transpose, Softmax, ReduceMax, TopK,
ArgMax, NonMaxSuppression, GatherND
```

官方模型包含 `ReduceMax`、`ArgMax` 和 `NonMaxSuppression`，不包含 `TopK` 和
`OneHot`。其中 ReduceMax、ArgMax、NMS、两个 FP32 GatherND、一个 INT64 GatherND
以及非空 INT64 Transpose/Slice 已由 Dayan 接管；Greater、NonZero、普通 Gather、
零宽分支及其余索引控制节点由 ORT 处理。实际算子数量和 assignment 以
[`yolov8n_official_operator_counts.md`](yolov8n_official_operator_counts.md) 为准。

手动优化模型保持 `[1,300,6]` 输出，用一个 `TopK(K=1)` 同时生成最高类别分数和
类别编号，不再包含 `ReduceMax` 和 `ArgMax`；官方模型保持不变。

选择 YOLOv8n 作为主适配模型的依据：

| 对比项 | YOLOv5n | YOLOv8n | YOLOv8 的进步及接入意义 |
|---|---|---|---|
| 检测框产生方式 | 使用预设 anchor，再预测调整量 | anchor-free，直接预测候选点到四条边的距离 | 不依赖预设框形状，代表较新的检测头设计 |
| 分类和框回归 | 在同一检测输出中集中预测 | 分类分支和框回归分支分开 | 两项任务可以分别提取和处理特征 |
| 框回归 | 直接预测框参数 | DFL 概率分布经过 Softmax 得到四边距离 | 能表达更细的连续距离，也可以验证 Dayan 的 Softmax 和张量变换能力 |
| 640×640 候选位置 | 通常 25200 个 | 通常 8400 个 | 进入 TopK/NMS 前的候选数量更少，但 DFL 本身增加了计算 |
| COCO 综合检测分数（mAP50-95，越高越好） | 28.0 | 37.3 | 在同一测试集上，YOLOv8n 对目标类别和框位置的综合检测效果更好 |
| 参数量 / FLOPs | 2.6M / 7.7B | 3.2M / 8.7B | YOLOv8n 并非更小，选择它不是因为计算量最低 |
| Dayan 算子覆盖 | 主要验证 Conv、SiLU、特征融合和 anchor 解码 | 还验证 DFL Softmax、更多张量整理和四边解码 | 更能检验 Plugin EP 对现代检测网络的覆盖能力 |
| 工程定位 | 成熟的传统兼容基线 | 当前主适配模型 | 先完成 YOLOv8n，再用 YOLOv5 验证向旧模型兼容 |

因此选择 YOLOv8n，不是因为版本号更大，也不是因为它在所有平台上都更快，而是
因为它在模型成熟度、模拟器可承受规模、检测精度和算子代表性之间更均衡。
YOLOv5 的结构更简单，仍适合作为后续兼容性验证，但如果只支持 YOLOv5，无法覆盖
YOLOv8 DFL 引入的 `Softmax` 和四边距离解码。

`mAP50-95` 不是“有多少张图片识别正确”的比例，而是同时检查目标类别和检测框位置，
并在多种框重合度要求下取平均得到的综合分数。28.0 提升到 37.3，表示该分数提高了
9.3 分；分数越高，整体检测效果越好。

数据来源：
[YOLOv8 官方结构说明](https://docs.ultralytics.com/models/yolov8/)、
[YOLOv5 与 YOLO11 官方对比](https://docs.ultralytics.com/compare/yolo11-vs-yolov5/)、
[YOLOv8 与 YOLO11 官方对比](https://docs.ultralytics.com/compare/yolov8-vs-yolo11/)。
