#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def values(shape: list[int], salt: int) -> np.ndarray:
    count = int(np.prod(shape))
    data = ((np.arange(count, dtype=np.int64) * 17 + salt * 11) % 41 - 20)
    return (data.astype(np.float32) * np.float32(0.125)).reshape(shape)


def make_case(directory: Path, op: str, shapes: list[list[int]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    names = ["input"] if len(shapes) == 1 else ["lhs", "rhs"]
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in zip(names, shapes)
    ]
    arrays = [values(shape, index + 1) for index, shape in enumerate(shapes)]
    if op in {"Add", "Mul", "Sub", "Div"}:
        output_shape = list(np.broadcast_shapes(*[array.shape for array in arrays]))
    else:
        output_shape = shapes[0]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    node = helper.make_node(op, names, ["output"], name=f"rvnpu_test_{op.lower()}")
    graph = helper.make_graph([node], f"rvnpu_{op.lower()}_test", inputs, [output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime-rvnpu",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, directory / "model.onnx")

    for name, array in zip(names, arrays):
        array.tofile(directory / f"{name}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=["add", "mul", "sub", "div", "sigmoid", "silu"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.op == "add":
        make_case(args.output_dir / "supported", "Add", [[2, 3, 5], [2, 3, 5]])
    elif args.op == "mul":
        make_case(
            args.output_dir / "supported",
            "Mul",
            [[1, 3, 1, 1], [1, 3, 4, 5]],
        )
    elif args.op == "sub":
        make_case(
            args.output_dir / "supported",
            "Sub",
            [[1, 2, 16], [1, 2, 16]],
        )
    elif args.op == "div":
        make_case(
            args.output_dir / "supported",
            "Div",
            [[1, 2, 16], [1]],
        )
    elif args.op == "sigmoid":
        make_case(args.output_dir / "supported", "Sigmoid", [[2, 3, 5]])
    else:
        shape = [1, 2, 3, 16]
        for directory_name, mul_inputs in (
            ("supported", ["input", "sigmoid"]),
            ("supported_swapped", ["sigmoid", "input"]),
        ):
            directory = args.output_dir / directory_name
            directory.mkdir(parents=True, exist_ok=True)
            input_info = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, shape
            )
            output_info = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, shape
            )
            nodes = [
                helper.make_node(
                    "Sigmoid",
                    ["input"],
                    ["sigmoid"],
                    name="rvnpu_test_sigmoid",
                ),
                helper.make_node(
                    "Mul", mul_inputs, ["output"], name="rvnpu_test_mul"
                ),
            ]
            graph = helper.make_graph(
                nodes, "rvnpu_silu_fusion_test", [input_info], [output_info]
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 13)],
                producer_name="onnxruntime-rvnpu",
            )
            model.ir_version = 10
            onnx.checker.check_model(model)
            onnx.save(
                shape_inference.infer_shapes(model), directory / "model.onnx"
            )
            values(shape, 1).tofile(directory / "input.bin")


if __name__ == "__main__":
    main()
