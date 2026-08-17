#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx


def tensor_shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("") != 13:
        raise ValueError(f"YOLOv8n model must use opset 13, got {opsets.get('')}")
    if len(model.graph.input) != 1 or tensor_shape(model.graph.input[0]) != [
        1,
        3,
        640,
        640,
    ]:
        raise ValueError("YOLOv8n model must have static input [1,3,640,640]")
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("YOLOv8n input must be FP32")

    expected_output = [1, 300, 6]
    if len(model.graph.output) != 1 or tensor_shape(model.graph.output[0]) != expected_output:
        raise ValueError(f"official YOLOv8n output must be {expected_output}")
    if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("YOLOv8n output must be FP32")

    counts = Counter(node.op_type for node in model.graph.node)
    for name in ("ReduceMax", "ArgMax", "NonMaxSuppression"):
        if counts[name] != 1:
            raise ValueError(f"official YOLOv8n must contain one {name}")
    if counts["TopK"]:
        raise ValueError("official YOLOv8n must not contain TopK")

    if args.output_dir is None:
        return
    count = 1 * 3 * 640 * 640
    input_data = ((np.arange(count, dtype=np.uint32) * 17 + 11) % 256).astype(
        np.float32
    )
    input_data = (input_data / np.float32(255.0)).reshape(1, 3, 640, 640)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_data.tofile(args.output_dir / f"{model.graph.input[0].name}.bin")


if __name__ == "__main__":
    main()
