#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference


def tensor_shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    if len(model.graph.input) != 1 or tensor_shape(model.graph.input[0]) != [
        1,
        3,
        640,
        640,
    ]:
        raise ValueError("YOLOv8n model must have static input [1,3,640,640]")
    if len(model.graph.output) != 1:
        raise ValueError("YOLOv8n test expects one output without embedded NMS")

    input_name = model.graph.input[0].name
    count = 1 * 3 * 640 * 640
    input_data = (
        (np.arange(count, dtype=np.uint32) * 17 + 11) % 256
    ).astype(np.float32)
    input_data = (input_data / np.float32(255.0)).reshape(1, 3, 640, 640)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_data.tofile(args.output_dir / f"{input_name}.bin")


if __name__ == "__main__":
    main()
