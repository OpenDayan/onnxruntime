#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def make_case(directory: Path, center_point_box: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    boxes = np.array(
        [[[0.0, 0.0, 2.0, 2.0],
          [0.1, 0.1, 2.1, 2.1],
          [3.0, 3.0, 4.0, 4.0],
          [5.0, 0.0, 6.0, 1.0]]],
        dtype=np.float32,
    )
    scores = np.array(
        [[[0.9, 0.8, 0.7, 0.6], [0.1, 0.95, 0.5, 0.85]]],
        dtype=np.float32,
    )
    max_output = np.array([2], dtype=np.int64)
    iou_threshold = np.array([0.5], dtype=np.float32)
    score_threshold = np.array([0.0], dtype=np.float32)
    node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        name="rvnpu_test_nms",
        center_point_box=center_point_box,
    )
    graph = helper.make_graph(
        [node],
        "rvnpu_nms_test",
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2, 4]),
        ],
        [
            helper.make_tensor_value_info(
                "selected_indices", TensorProto.INT64, [None, 3]
            )
        ],
        [
            numpy_helper.from_array(max_output, "max_output"),
            numpy_helper.from_array(iou_threshold, "iou_threshold"),
            numpy_helper.from_array(score_threshold, "score_threshold"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime-rvnpu",
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
            name="rvnpu_topk", axis=-1, largest=1, sorted=1
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
            name="rvnpu_nms",
            center_point_box=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "rvnpu_topk_nms_test",
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
        producer_name="onnxruntime-rvnpu",
    )
    model.ir_version = 10
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    boxes.tofile(directory / "boxes.bin")
    scores.tofile(directory / "scores.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_case(args.output_dir / "supported", 0)
    make_topk_pipeline(args.output_dir / "topk_pipeline")


if __name__ == "__main__":
    main()
