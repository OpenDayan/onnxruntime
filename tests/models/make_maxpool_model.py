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
        [node], "dayan_maxpool_test", [graph_input], [graph_output], initializers
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


def make_maxpool_cases(output_dir: Path) -> None:
    supported = helper.make_node(
        "MaxPool",
        ["input"],
        ["output"],
        name="dayan_test_maxpool",
        ceil_mode=0,
        dilations=[1, 1],
        kernel_shape=[5, 5],
        pads=[2, 2, 2, 2],
        strides=[1, 1],
    )
    save_case(output_dir / "supported", supported, [1, 3, 8, 9], [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_maxpool_cases(args.output_dir)


if __name__ == "__main__":
    main()
