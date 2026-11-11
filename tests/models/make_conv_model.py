#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def values(shape: list[int], salt: int) -> np.ndarray:
    count = int(np.prod(shape))
    data = ((np.arange(count, dtype=np.int64) * 17 + salt * 13) % 37 - 18)
    return (data.astype(np.float32) * np.float32(0.0625)).reshape(shape)


def save_case(
    directory: Path,
    node: onnx.NodeProto,
    input_shape: list[int],
    initializers: list[onnx.TensorProto],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    graph_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    graph_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, None, None, None]
    )
    graph = helper.make_graph(
        [node], "dayan_conv_test", [graph_input], [graph_output], initializers
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")

    input_data = values(input_shape, 1)
    input_data.tofile(directory / "input.bin")


def make_conv_cases(output_dir: Path) -> None:
    weight = values([5, 3, 3, 3], 2)
    bias = values([5], 3)
    supported = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        name="dayan_test_conv",
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )
    save_case(
        output_dir / "supported",
        supported,
        [1, 3, 15, 17],
        [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_conv_cases(args.output_dir)


if __name__ == "__main__":
    main()
