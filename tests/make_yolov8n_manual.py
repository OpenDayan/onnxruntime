#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


RAW_SHAPE = [1, 84, 8400]
OUTPUT_SHAPE = [1, 300, 6]
PREFIX = "rvnpu_manual_"


def value_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    return {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in values
    }


def find_raw_output(model: onnx.ModelProto) -> tuple[int, str]:
    inferred = shape_inference.infer_shapes(model)
    shapes = value_shapes(inferred)
    matches = [
        (index, output)
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if shapes.get(output) == RAW_SHAPE
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one YOLOv8n raw output {RAW_SHAPE}, got {matches}")
    return matches[0]


def initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, PREFIX + name)


def make_manual_model(official: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(official)
    raw_node_index, raw_output = find_raw_output(model)
    del model.graph.node[raw_node_index + 1 :]

    model.graph.initializer.extend(
        [
            initializer("channel_split", np.array([4, 80], dtype=np.int64)),
            initializer("topk_k", np.array([1], dtype=np.int64)),
            initializer("class_offset", np.full((1, 1, 4), 7680.0, dtype=np.float32)),
            initializer("max_output", np.array([300], dtype=np.int64)),
            initializer("iou_threshold", np.array([0.7], dtype=np.float32)),
            initializer("score_threshold", np.array([0.25], dtype=np.float32)),
            initializer("box_index_columns", np.array([0, 2], dtype=np.int64)),
            initializer("output_rows", np.array(300, dtype=np.int64)),
            initializer("shape_row", np.array(0, dtype=np.int64)),
            initializer("unsqueeze_axis0", np.array([0], dtype=np.int64)),
            initializer("pads_begin", np.array([0, 0], dtype=np.int64)),
            initializer("pad_column", np.array([0], dtype=np.int64)),
        ]
    )

    def name(value: str) -> str:
        return PREFIX + value

    model.graph.node.extend(
        [
            helper.make_node(
                "Split",
                [raw_output, name("channel_split")],
                [name("boxes_1x4xN"), name("scores_1x80xN")],
                axis=1,
                name=name("SplitRaw"),
            ),
            helper.make_node(
                "Transpose",
                [name("boxes_1x4xN")],
                [name("boxes")],
                perm=[0, 2, 1],
                name=name("TransposeBoxes"),
            ),
            helper.make_node(
                "Transpose",
                [name("scores_1x80xN")],
                [name("scores_1xNx80")],
                perm=[0, 2, 1],
                name=name("TransposeScores"),
            ),
            helper.make_node(
                "TopK",
                [name("scores_1xNx80"), name("topk_k")],
                [name("best_scores"), name("best_classes")],
                axis=-1,
                largest=1,
                sorted=1,
                name=name("TopKClass"),
            ),
            helper.make_node(
                "Transpose",
                [name("best_scores")],
                [name("nms_scores")],
                perm=[0, 2, 1],
                name=name("TransposeBestScores"),
            ),
            helper.make_node(
                "Cast",
                [name("best_classes")],
                [name("best_classes_fp32")],
                to=TensorProto.FLOAT,
                name=name("CastClasses"),
            ),
            helper.make_node(
                "Mul",
                [name("best_classes_fp32"), name("class_offset")],
                [name("class_offsets")],
                name=name("ClassOffsets"),
            ),
            helper.make_node(
                "Add",
                [name("boxes"), name("class_offsets")],
                [name("nms_boxes")],
                name=name("OffsetBoxes"),
            ),
            helper.make_node(
                "NonMaxSuppression",
                [
                    name("nms_boxes"),
                    name("nms_scores"),
                    name("max_output"),
                    name("iou_threshold"),
                    name("score_threshold"),
                ],
                [name("selected_indices")],
                center_point_box=0,
                name=name("NonMaxSuppression"),
            ),
            helper.make_node(
                "Gather",
                [name("selected_indices"), name("box_index_columns")],
                [name("box_indices")],
                axis=1,
                name=name("BoxIndices"),
            ),
            helper.make_node(
                "GatherND",
                [name("boxes"), name("box_indices")],
                [name("selected_boxes")],
                name=name("SelectedBoxes"),
            ),
            helper.make_node(
                "GatherND",
                [name("best_scores"), name("box_indices")],
                [name("selected_scores")],
                name=name("SelectedScores"),
            ),
            helper.make_node(
                "GatherND",
                [name("best_classes"), name("box_indices")],
                [name("selected_classes")],
                name=name("SelectedClasses"),
            ),
            helper.make_node(
                "Cast",
                [name("selected_classes")],
                [name("selected_classes_fp32")],
                to=TensorProto.FLOAT,
                name=name("CastSelectedClasses"),
            ),
            helper.make_node(
                "Concat",
                [
                    name("selected_boxes"),
                    name("selected_scores"),
                    name("selected_classes_fp32"),
                ],
                [name("detections")],
                axis=1,
                name=name("PackDetections"),
            ),
            helper.make_node(
                "Shape",
                [name("detections")],
                [name("detection_shape")],
                name=name("DetectionShape"),
            ),
            helper.make_node(
                "Gather",
                [name("detection_shape"), name("shape_row")],
                [name("detection_rows")],
                axis=0,
                name=name("DetectionRows"),
            ),
            helper.make_node(
                "Sub",
                [name("output_rows"), name("detection_rows")],
                [name("padding_rows")],
                name=name("PaddingRows"),
            ),
            helper.make_node(
                "Unsqueeze",
                [name("padding_rows"), name("unsqueeze_axis0")],
                [name("padding_rows_1d")],
                name=name("UnsqueezePaddingRows"),
            ),
            helper.make_node(
                "Concat",
                [name("pads_begin"), name("padding_rows_1d"), name("pad_column")],
                [name("pads")],
                axis=0,
                name=name("MakePads"),
            ),
            helper.make_node(
                "Pad",
                [name("detections"), name("pads")],
                [name("padded_detections")],
                mode="constant",
                name=name("PadDetections"),
            ),
            helper.make_node(
                "Unsqueeze",
                [name("padded_detections"), name("unsqueeze_axis0")],
                [name("output")],
                name=name("AddBatchDimension"),
            ),
        ]
    )

    del model.graph.output[:]
    model.graph.output.extend(
        [helper.make_tensor_value_info(name("output"), TensorProto.FLOAT, OUTPUT_SHAPE)]
    )
    del model.graph.value_info[:]

    used = {input_name for node in model.graph.node for input_name in node.input}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = onnx.load(args.official)
    manual = make_manual_model(official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(manual, args.output)


if __name__ == "__main__":
    main()
