#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def save_model(directory: Path, graph: onnx.GraphProto) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime-rvnpu",
    )
    model.ir_version = 10
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")


def make_argmax(root: Path) -> None:
    directory = root / "argmax"
    data = np.array(
        [[1.0, 5.0, 5.0, 2.0], [-1.0, 3.0, 2.0, 4.0]],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "ArgMax",
                ["input"],
                ["output"],
                name="rvnpu_test_argmax",
                axis=-1,
                keepdims=0,
                select_last_index=0,
            )
        ],
        "rvnpu_argmax_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [2])],
    )
    save_model(directory, graph)
    data.tofile(directory / "input.bin")


def make_gather_nd(root: Path) -> None:
    directory = root / "gather_nd_int64"
    data = np.array([17, 31, 43, 59, 61, 73, 89, 97], dtype=np.int64)
    indices = np.array([[6], [1], [7], [3]], dtype=np.int64)
    graph = helper.make_graph(
        [
            helper.make_node(
                "GatherND",
                ["data", "indices"],
                ["output"],
                name="rvnpu_test_gather_nd_int64",
                batch_dims=0,
            )
        ],
        "rvnpu_gather_nd_int64_test",
        [
            helper.make_tensor_value_info("data", TensorProto.INT64, [8]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [4, 1]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [4])],
    )
    save_model(directory, graph)
    data.tofile(directory / "data.bin")
    indices.tofile(directory / "indices.bin")


def make_transpose_int64(root: Path) -> None:
    directory = root / "transpose_int64"
    data = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose",
                ["data"],
                ["output"],
                name="rvnpu_test_transpose_int64",
                perm=[1, 0],
            )
        ],
        "rvnpu_transpose_int64_test",
        [helper.make_tensor_value_info("data", TensorProto.INT64, [3, 2])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [2, 3])],
    )
    save_model(directory, graph)
    data.tofile(directory / "data.bin")


def make_transpose_int64_vector(root: Path) -> None:
    directory = root / "transpose_int64_vector"
    data = np.arange(45, dtype=np.int64).reshape(1, 45)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose",
                ["data"],
                ["output"],
                name="rvnpu_test_transpose_int64_vector",
                perm=[1, 0],
            )
        ],
        "rvnpu_transpose_int64_vector_test",
        [helper.make_tensor_value_info("data", TensorProto.INT64, [1, 45])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [45, 1])],
    )
    save_model(directory, graph)
    data.tofile(directory / "data.bin")


def make_slice_int64(root: Path) -> None:
    directory = root / "slice_int64"
    data = np.arange(12, dtype=np.int64).reshape(6, 2)
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "steps"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                ["data", "starts", "ends", "axes", "steps"],
                ["output"],
                name="rvnpu_test_slice_int64",
            )
        ],
        "rvnpu_slice_int64_test",
        [helper.make_tensor_value_info("data", TensorProto.INT64, [6, 2])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [4, 2])],
        initializers,
    )
    save_model(directory, graph)
    data.tofile(directory / "data.bin")


def make_slice_int64_reverse(root: Path) -> None:
    directory = root / "slice_int64_reverse"
    data = np.arange(10, dtype=np.int64).reshape(5, 2)
    initializers = [
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "starts"),
        numpy_helper.from_array(
            np.array([np.iinfo(np.int64).min + 1], dtype=np.int64), "ends"
        ),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "steps"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                ["data", "starts", "ends", "axes", "steps"],
                ["output"],
                name="rvnpu_test_slice_int64_reverse",
            )
        ],
        "rvnpu_slice_int64_reverse_test",
        [helper.make_tensor_value_info("data", TensorProto.INT64, [5, 2])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [5, 2])],
        initializers,
    )
    save_model(directory, graph)
    data.tofile(directory / "data.bin")


def make_empty_postprocess(root: Path) -> None:
    directory = root / "empty_postprocess"
    scores = np.zeros(8, dtype=np.float32)
    threshold = np.array(0.25, dtype=np.float32)
    data = np.arange(8, dtype=np.float32)
    divisor = np.array(640.0, dtype=np.float32)
    nodes = [
        helper.make_node(
            "Greater", ["scores", "threshold"], ["mask"], name="empty_greater"
        ),
        helper.make_node(
            "NonZero", ["mask"], ["coordinates"], name="empty_nonzero"
        ),
        helper.make_node(
            "Transpose",
            ["coordinates"],
            ["indices"],
            name="empty_transpose",
            perm=[1, 0],
        ),
        helper.make_node(
            "GatherND", ["data", "indices"], ["selected"], name="empty_gather_nd"
        ),
        helper.make_node(
            "Div", ["selected", "divisor"], ["normalized"], name="empty_div"
        ),
        helper.make_node(
            "Unsqueeze",
            ["normalized", "column_axes"],
            ["column"],
            name="empty_column",
        ),
        helper.make_node(
            "Concat",
            ["column", "column", "column", "column"],
            ["boxes_2d"],
            name="empty_concat",
            axis=1,
        ),
        helper.make_node(
            "Slice",
            [
                "boxes_2d",
                "slice_starts",
                "slice_ends",
                "slice_axes",
                "slice_steps",
            ],
            ["boxes_slice"],
            name="empty_slice",
        ),
        helper.make_node(
            "Unsqueeze",
            ["boxes_slice", "boxes_axes"],
            ["boxes"],
            name="empty_boxes",
        ),
        helper.make_node(
            "Unsqueeze",
            ["normalized", "scores_axes"],
            ["nms_scores"],
            name="empty_scores",
        ),
        helper.make_node(
            "NonMaxSuppression",
            [
                "boxes",
                "nms_scores",
                "max_output",
                "iou_threshold",
                "score_threshold",
            ],
            ["output"],
            name="empty_nms",
            center_point_box=0,
        ),
    ]
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "column_axes"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "boxes_axes"),
        numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "scores_axes"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_starts"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), "slice_ends"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_steps"),
        numpy_helper.from_array(np.array([10], dtype=np.int64), "max_output"),
        numpy_helper.from_array(
            np.array([0.5], dtype=np.float32), "iou_threshold"
        ),
        numpy_helper.from_array(
            np.array([0.0], dtype=np.float32), "score_threshold"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "rvnpu_empty_postprocess_test",
        [
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [8]),
            helper.make_tensor_value_info("threshold", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [8]),
            helper.make_tensor_value_info("divisor", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [None, 3])],
        initializers,
    )
    save_model(directory, graph)
    scores.tofile(directory / "scores.bin")
    threshold.tofile(directory / "threshold.bin")
    data.tofile(directory / "data.bin")
    divisor.tofile(directory / "divisor.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_argmax(args.output_dir)
    make_gather_nd(args.output_dir)
    make_transpose_int64(args.output_dir)
    make_transpose_int64_vector(args.output_dir)
    make_slice_int64(args.output_dir)
    make_slice_int64_reverse(args.output_dir)
    make_empty_postprocess(args.output_dir)


if __name__ == "__main__":
    main()
