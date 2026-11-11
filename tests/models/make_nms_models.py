#!/usr/bin/env python3
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# onnxruntime is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def make_case(directory: Path, center_point_box: int, constant_node: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    boxes = np.array(
        [[[0.0, 0.0, 2.0, 2.0],
          [0.1, 0.1, 2.1, 2.1],
          [3.0, 3.0, 4.0, 4.0],
          [5.0, 0.0, 6.0, 1.0]]],
        dtype=np.float32,
    )
    if center_point_box == 1:
        y1, x1, y2, x2 = [boxes[..., index] for index in range(4)]
        boxes = np.stack(
            ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1),
            axis=-1,
        ).astype(np.float32)
    scores = np.array(
        [[[0.9, 0.8, 0.7, 0.6], [0.1, 0.95, 0.5, 0.85]]],
        dtype=np.float32,
    )
    max_output = np.array(
        [np.iinfo(np.int64).max if constant_node else 2], dtype=np.int64
    )
    iou_threshold = np.array([0.5], dtype=np.float32)
    score_threshold = np.array([0.0], dtype=np.float32)
    nms = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        name="dayan_test_nms",
        center_point_box=center_point_box,
    )
    nodes = [nms]
    initializers = [
        numpy_helper.from_array(iou_threshold, "iou_threshold"),
        numpy_helper.from_array(score_threshold, "score_threshold"),
    ]
    if constant_node:
        nodes.insert(
            0,
            helper.make_node(
                "Constant",
                [],
                ["max_output"],
                value=numpy_helper.from_array(max_output),
            ),
        )
    else:
        initializers.append(numpy_helper.from_array(max_output, "max_output"))
    graph = helper.make_graph(
        nodes,
        "dayan_nms_test",
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2, 4]),
        ],
        [
            helper.make_tensor_value_info(
                "selected_indices", TensorProto.INT64, [None, 3]
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime",
    )
    model.ir_version = 10
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    boxes.tofile(directory / "boxes.bin")
    scores.tofile(directory / "scores.bin")


def make_topk_pipeline(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    boxes = np.array(
        [[
            [0.0, 0.0, 2.0, 2.0],
            [0.1, 0.1, 2.1, 2.1],
            [3.0, 3.0, 4.0, 4.0],
            [5.0, 0.0, 6.0, 1.0],
            [7.0, 0.0, 8.0, 1.0],
            [9.0, 0.0, 10.0, 1.0],
            [11.0, 0.0, 12.0, 1.0],
            [13.0, 0.0, 14.0, 1.0],
        ]],
        dtype=np.float32,
    )
    scores = np.array(
        [[0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.4, 0.3]], dtype=np.float32
    )
    initializers = [
        numpy_helper.from_array(np.array([4], dtype=np.int64), "k"),
        numpy_helper.from_array(np.array([1, 4, 4], dtype=np.int64), "box_shape"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "score_axes"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "max_output"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "iou_threshold"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), "score_threshold"),
    ]
    nodes = [
        helper.make_node(
            "TopK", ["scores", "k"], ["top_scores", "top_indices"],
            name="dayan_topk", axis=-1, largest=1, sorted=1
        ),
        helper.make_node(
            "Gather", ["boxes", "top_indices"], ["gathered_boxes"],
            name="cpu_gather", axis=1
        ),
        helper.make_node(
            "Reshape", ["gathered_boxes", "box_shape"], ["nms_boxes"],
            name="cpu_reshape"
        ),
        helper.make_node(
            "Unsqueeze", ["top_scores", "score_axes"], ["nms_scores"],
            name="cpu_unsqueeze"
        ),
        helper.make_node(
            "NonMaxSuppression",
            ["nms_boxes", "nms_scores", "max_output", "iou_threshold", "score_threshold"],
            ["selected_indices"],
            name="dayan_nms",
            center_point_box=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "dayan_topk_nms_test",
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 8, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8]),
        ],
        [
            helper.make_tensor_value_info(
                "selected_indices", TensorProto.INT64, [None, 3]
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime",
    )
    model.ir_version = 10
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    boxes.tofile(directory / "boxes.bin")
    scores.tofile(directory / "scores.bin")


def make_yolo_postprocess(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    num_boxes = 16
    raw = np.zeros((1, 84, num_boxes), dtype=np.float32)
    boxes = np.array(
        [[[1.0, 1.0, 2.0, 2.0],
          [1.1, 1.1, 2.0, 2.0],
          [3.5, 3.5, 1.0, 1.0],
          [0.5, 5.5, 1.0, 1.0],
          [7.5, 0.5, 1.0, 1.0],
          [9.5, 0.5, 1.0, 1.0],
          [11.5, 0.5, 1.0, 1.0],
          [13.5, 0.5, 1.0, 1.0],
          [15.5, 0.5, 1.0, 1.0],
          [17.5, 0.5, 1.0, 1.0],
          [19.5, 0.5, 1.0, 1.0],
          [21.5, 0.5, 1.0, 1.0],
          [23.5, 0.5, 1.0, 1.0],
          [25.5, 0.5, 1.0, 1.0],
          [27.5, 0.5, 1.0, 1.0],
          [29.5, 0.5, 1.0, 1.0]]],
        dtype=np.float32,
    )
    scores = np.zeros((1, 80, num_boxes), dtype=np.float32)
    scores[0, 0, :4] = [0.9, 0.8, 0.7, 0.1]
    scores[0, 17, 3:5] = [0.85, 0.6]
    raw[:, :4, :] = np.transpose(boxes, (0, 2, 1))
    raw[:, 4:, :] = scores

    initializers = [
        numpy_helper.from_array(np.array([4, 80], dtype=np.int64), "channel_split"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "max_output"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "iou_threshold"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "score_threshold"),
        numpy_helper.from_array(np.array([0, 2], dtype=np.int64), "box_index_columns"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "class_column"),
    ]
    nodes = [
        helper.make_node(
            "Split", ["raw_output", "channel_split"],
            ["boxes_1x4xN", "nms_scores"], axis=1,
        ),
        helper.make_node(
            "Transpose", ["boxes_1x4xN"], ["nms_boxes"], perm=[0, 2, 1],
        ),
        helper.make_node(
            "NonMaxSuppression",
            ["nms_boxes", "nms_scores", "max_output", "iou_threshold", "score_threshold"],
            ["selected_indices"], center_point_box=1,
        ),
        helper.make_node(
            "Gather", ["selected_indices", "box_index_columns"],
            ["box_indices"], axis=1,
        ),
        helper.make_node(
            "Gather", ["selected_indices", "class_column"],
            ["selected_classes"], axis=1,
        ),
        helper.make_node(
            "GatherND", ["nms_boxes", "box_indices"], ["selected_boxes"],
        ),
        helper.make_node(
            "GatherND", ["nms_scores", "selected_indices"],
            ["selected_scores"],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "dayan_yolo_postprocess_test",
        [helper.make_tensor_value_info("raw_output", TensorProto.FLOAT, [1, 84, num_boxes])],
        [
            helper.make_tensor_value_info("selected_boxes", TensorProto.FLOAT, [None, 4]),
            helper.make_tensor_value_info("selected_scores", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("selected_classes", TensorProto.INT64, [None, 1]),
        ],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime",
    )
    model.ir_version = 10
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    raw.tofile(directory / "raw_output.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_case(args.output_dir / "supported", 0)
    make_case(args.output_dir / "supported_center", 1, constant_node=True)
    make_topk_pipeline(args.output_dir / "topk_pipeline")
    make_yolo_postprocess(args.output_dir / "yolo_postprocess")


if __name__ == "__main__":
    main()
