#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def values(shape: list[int], salt: int = 1) -> np.ndarray:
    count = int(np.prod(shape))
    return (np.arange(count, dtype=np.float32) + np.float32(salt * 100)).reshape(shape)


def save_model(directory: Path, model: onnx.ModelProto, inputs: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(shape_inference.infer_shapes(model), directory / "model.onnx")
    for name, array in inputs.items():
        array.tofile(directory / f"{name}.bin")


def make_concat(directory: Path) -> None:
    a_shape, b_shape = [1, 2, 2, 16], [1, 1, 2, 16]
    graph = helper.make_graph(
        [helper.make_node("Concat", ["a", "b"], ["output"], axis=1)],
        "dayan_concat_test",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, b_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 16])],
    )
    save_model(directory, helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
               {"a": values(a_shape, 1), "b": values(b_shape, 2)})


def make_split(directory: Path) -> None:
    shape = [1, 4, 2, 16]
    split = helper.make_tensor("split", TensorProto.INT64, [2], [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Split", ["input", "split"], ["first", "unused"], axis=1)],
        "dayan_split_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("first", TensorProto.FLOAT, [1, 2, 2, 16])],
        [split],
    )
    save_model(directory, helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
               {"input": values(shape)})


def make_slice(directory: Path) -> None:
    shape = [1, 4, 16]
    initializers = [
        helper.make_tensor("starts", TensorProto.INT64, [1], [1]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [3]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
    ]
    graph = helper.make_graph(
        [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["output"])],
        "dayan_slice_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 16])],
        initializers,
    )
    save_model(directory, helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
               {"input": values(shape)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["concat", "split", "slice"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {"concat": make_concat, "split": make_split, "slice": make_slice}[args.op](
        args.output_dir / "supported"
    )


if __name__ == "__main__":
    main()
