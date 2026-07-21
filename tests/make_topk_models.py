#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def make_case(directory: Path, rows: int, columns: int, k: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [rows, columns]
    )
    values_info = helper.make_tensor_value_info(
        "values", TensorProto.FLOAT, [rows, k]
    )
    indices_info = helper.make_tensor_value_info(
        "indices", TensorProto.INT64, [rows, k]
    )
    k_value = np.array([k], dtype=np.int64)
    node = helper.make_node(
        "TopK",
        ["input", "k"],
        ["values", "indices"],
        name="rvnpu_test_topk",
        axis=-1,
        largest=1,
        sorted=1,
    )
    graph = helper.make_graph(
        [node],
        "rvnpu_topk_test",
        [input_info],
        [values_info, indices_info],
        [numpy_helper.from_array(k_value, "k")],
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

    flat = (np.arange(rows * columns, dtype=np.int64) * 7) % 11
    input_data = (flat.astype(np.float32) * np.float32(0.125)).reshape(
        rows, columns
    )
    input_data.tofile(directory / "input.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_case(args.output_dir / "supported", 3, 17, 5)


if __name__ == "__main__":
    main()
