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

"""Generate one standard ONNX CNN and deterministic input data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def deterministic_values(shape: tuple[int, ...], scale: float, offset: int) -> np.ndarray:
    count = int(np.prod(shape))
    values = ((np.arange(count, dtype=np.int32) * 17 + offset) % 31 - 15).astype(np.float32)
    return (values * np.float32(scale)).reshape(shape)


def build_model() -> onnx.ModelProto:
    # Sparse selector kernels keep the integration model's values exactly
    # representable while still exercising Conv, Relu and MaxPool scheduling.
    conv_weight = np.zeros((2, 1, 3, 3), dtype=np.float32)
    conv_weight[0, 0, 1, 1] = np.float32(1.0)
    conv_weight[1, 0, 0, 0] = np.float32(0.5)
    conv_bias = np.zeros((2,), dtype=np.float32)
    matmul_weight = deterministic_values((32, 32), 1.0 / 32.0, 7)

    nodes = [
        helper.make_node("Conv", ["input", "conv_weight", "conv_bias"], ["conv_out"], name="conv"),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu"),
        helper.make_node(
            "MaxPool", ["relu_out"], ["pool_out"], name="maxpool", kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node("Flatten", ["pool_out"], ["flat_out"], name="flatten", axis=1),
        helper.make_node("MatMul", ["flat_out", "matmul_weight"], ["output"], name="matmul"),
    ]
    graph = helper.make_graph(
        nodes,
        "dayan_tiny_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 10, 10])],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32]),
            helper.make_tensor_value_info("flat_out", TensorProto.FLOAT, [1, 32]),
        ],
        [
            numpy_helper.from_array(conv_weight, "conv_weight"),
            numpy_helper.from_array(conv_bias, "conv_bias"),
            numpy_helper.from_array(matmul_weight, "matmul_weight"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnxruntime",
        ir_version=10,
    )
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    expected = ["Conv", "Relu", "MaxPool", "Flatten", "MatMul"]
    actual = [node.op_type for node in inferred.graph.node]
    if actual != expected:
        raise RuntimeError(f"unexpected graph nodes: {actual}")
    return inferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model_path = args.output_dir / "tiny_cnn.onnx"
    onnx.save(model, model_path)

    input_data = deterministic_values((1, 1, 10, 10), 1.0 / 16.0, 11)
    input_data.astype(np.float32).tofile(args.output_dir / "input.bin")
    print(model_path)


if __name__ == "__main__":
    main()
