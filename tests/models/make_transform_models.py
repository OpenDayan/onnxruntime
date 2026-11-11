#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def save(directory: Path, graph: onnx.GraphProto, data: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(shape_inference.infer_shapes(model), directory / "model.onnx")
    data.astype(np.float32).tofile(directory / "input.bin")


def make_resize(directory: Path) -> None:
    input_shape, output_shape = [1, 2, 4, 4], [1, 2, 8, 8]
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("Resize", ["input", "", "scales"], ["output"],
                          mode="nearest", coordinate_transformation_mode="asymmetric",
                          nearest_mode="floor")],
        "dayan_resize_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        [scales],
    )
    save(directory, graph, np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape))


def make_transpose(directory: Path) -> None:
    input_shape, output_shape = [1, 84, 32], [1, 32, 84]
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["input"], ["output"], perm=[0, 2, 1])],
        "dayan_transpose_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    save(directory, graph, np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape))


def make_reduce_max(directory: Path) -> None:
    input_shape, output_shape = [1, 32, 80], [1, 32]
    graph = helper.make_graph(
        [helper.make_node("ReduceMax", ["input"], ["output"], axes=[-1], keepdims=0)],
        "dayan_reduce_max_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    data = ((np.arange(np.prod(input_shape), dtype=np.int64) % 97) - 48).astype(np.float32)
    save(directory, graph, data.reshape(input_shape))


def make_softmax(directory: Path) -> None:
    shape = [1, 16, 2, 16]
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
        "dayan_softmax_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    data = ((np.arange(np.prod(shape), dtype=np.int64) % 13 - 6) / 4.0).astype(np.float32).reshape(shape)
    save(directory, graph, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op", choices=["resize", "transpose", "reduce_max", "softmax"], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {
        "resize": make_resize,
        "transpose": make_transpose,
        "reduce_max": make_reduce_max,
        "softmax": make_softmax,
    }[args.op](
        args.output_dir / "supported"
    )


if __name__ == "__main__":
    main()
