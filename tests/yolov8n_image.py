#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
from PIL import Image, ImageDraw


def tensor_shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def class_names(model: onnx.ModelProto) -> dict[int, str]:
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    value = metadata.get("names", "")
    if not value:
        return {}
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("ONNX metadata 'names' must be a dictionary")
    return {int(key): str(name) for key, name in parsed.items()}


def prepare(args: argparse.Namespace) -> None:
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise ValueError("YOLOv8n model must have one input")
    shape = tensor_shape(model.graph.input[0])
    if shape != [1, 3, 640, 640]:
        raise ValueError(f"YOLOv8n input must be [1,3,640,640], got {shape}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"cannot read image: {args.image}")
    original_height, original_width = image.shape[:2]
    input_height, input_width = shape[2], shape[3]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if letterboxed.shape[:2] != (input_height, input_width):
        raise RuntimeError("letterbox produced an unexpected image size")

    input_data = letterboxed[:, :, ::-1].transpose(2, 0, 1)
    input_data = np.ascontiguousarray(input_data, dtype=np.float32) / np.float32(255.0)
    input_data = input_data[np.newaxis, ...]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_name = model.graph.input[0].name
    input_data.tofile(args.output_dir / f"{input_name}.bin")
    metadata = {
        "image_path": str(args.image.resolve()),
        "input_name": input_name,
        "original_width": original_width,
        "original_height": original_height,
        "input_width": input_width,
        "input_height": input_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale": scale,
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
        "pad_x": left,
        "pad_y": top,
        "class_names": class_names(model),
    }
    (args.output_dir / "image_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"image={args.image.name} original={original_width}x{original_height} "
        f"input={input_width}x{input_height}"
    )


def xyxy_detection(detection: dict, metadata: dict, box_format: str) -> dict:
    first, second, third, fourth = detection["box"]
    if box_format == "xyxy":
        input_x1, input_y1, input_x2, input_y2 = first, second, third, fourth
    elif box_format == "cxcywh":
        input_x1 = first - third / 2
        input_y1 = second - fourth / 2
        input_x2 = first + third / 2
        input_y2 = second + fourth / 2
    else:
        raise ValueError(f"unsupported box format: {box_format}")
    scale_x = float(metadata["scale_x"])
    scale_y = float(metadata["scale_y"])
    pad_x = float(metadata["pad_x"])
    pad_y = float(metadata["pad_y"])
    image_width = float(metadata["original_width"])
    image_height = float(metadata["original_height"])
    x1 = np.clip((input_x1 - pad_x) / scale_x, 0, image_width)
    y1 = np.clip((input_y1 - pad_y) / scale_y, 0, image_height)
    x2 = np.clip((input_x2 - pad_x) / scale_x, 0, image_width)
    y2 = np.clip((input_y2 - pad_y) / scale_y, 0, image_height)
    return {
        "class_id": int(detection["class_id"]),
        "score": float(detection["score"]),
        "box": [float(x1), float(y1), float(x2), float(y2)],
    }


def box_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(cpu: list[dict], rvnpu: list[dict], threshold: float) -> dict:
    unmatched_rvnpu = set(range(len(rvnpu)))
    matches = []
    unmatched_cpu = []
    for cpu_index, cpu_detection in enumerate(cpu):
        candidates = [
            (box_iou(cpu_detection["box"], rvnpu[index]["box"]), index)
            for index in unmatched_rvnpu
            if rvnpu[index]["class_id"] == cpu_detection["class_id"]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index < 0 or best_iou < threshold:
            unmatched_cpu.append(cpu_index)
            continue
        unmatched_rvnpu.remove(best_index)
        matches.append(
            {
                "cpu_index": cpu_index,
                "rvnpu_index": best_index,
                "class_id": cpu_detection["class_id"],
                "iou": best_iou,
                "score_abs_diff": abs(
                    cpu_detection["score"] - rvnpu[best_index]["score"]
                ),
            }
        )
    ious = [match["iou"] for match in matches]
    score_diffs = [match["score_abs_diff"] for match in matches]
    return {
        "cpu_count": len(cpu),
        "rvnpu_count": len(rvnpu),
        "matched": len(matches),
        "unmatched_cpu": unmatched_cpu,
        "unmatched_rvnpu": sorted(unmatched_rvnpu),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": min(ious, default=0.0),
        "iou_at_least_0_95": sum(value >= 0.95 for value in ious),
        "max_score_abs_diff": max(score_diffs, default=0.0),
        "matches": matches,
    }


def draw_detections(image: Image.Image, detections: list[dict], names: dict, color) -> None:
    drawer = ImageDraw.Draw(image)
    for detection in detections:
        x1, y1, x2, y2 = [round(value) for value in detection["box"]]
        class_id = detection["class_id"]
        label = f"{names.get(str(class_id), str(class_id))} {detection['score']:.2f}"
        drawer.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_y = max(0, y1 - 14)
        text_box = drawer.textbbox((x1, text_y), label)
        drawer.rectangle(text_box, fill=color)
        drawer.text((x1, text_y), label, fill=(255, 255, 255))


def draw(args: argparse.Namespace) -> None:
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    image = Image.open(metadata["image_path"]).convert("RGB")

    box_format = results.get("box_format", "cxcywh")
    cpu = [xyxy_detection(item, metadata, box_format) for item in results["cpu"]]
    rvnpu = [
        xyxy_detection(item, metadata, box_format)
        for item in results.get("rvnpu", [])
    ]
    comparison = (
        match_detections(cpu, rvnpu, args.match_iou)
        if "rvnpu" in results
        else {"cpu_count": len(cpu)}
    )
    comparison["cpu"] = cpu
    comparison["rvnpu"] = rvnpu

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(metadata["image_path"]).stem
    names = metadata.get("class_names", {})
    cpu_image = image.copy()
    draw_detections(cpu_image, cpu, names, (0, 180, 0))
    cpu_image.save(args.output_dir / f"{stem}_cpu.jpg", quality=95)
    if "rvnpu" in results:
        rvnpu_image = image.copy()
        overlay_image = image.copy()
        draw_detections(rvnpu_image, rvnpu, names, (255, 0, 0))
        draw_detections(overlay_image, cpu, names, (0, 180, 0))
        draw_detections(overlay_image, rvnpu, names, (255, 0, 0))
        rvnpu_image.save(args.output_dir / f"{stem}_rvnpu.jpg", quality=95)
        overlay_image.save(args.output_dir / f"{stem}_overlay.jpg", quality=95)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "rvnpu" in results:
        print(
            f"detections cpu={comparison['cpu_count']} "
            f"rvnpu={comparison['rvnpu_count']} matched={comparison['matched']} "
            f"mean_iou={comparison['mean_iou']:.6g} "
            f"max_score_abs_diff={comparison['max_score_abs_diff']:.6g}"
        )
    else:
        print(f"detections cpu={comparison['cpu_count']}")
    print(f"images={args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--model", type=Path, required=True)
    prepare_parser.add_argument("--image", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.set_defaults(function=prepare)

    draw_parser = subparsers.add_parser("draw")
    draw_parser.add_argument("--metadata", type=Path, required=True)
    draw_parser.add_argument("--results", type=Path, required=True)
    draw_parser.add_argument("--output-dir", type=Path, required=True)
    draw_parser.add_argument("--match-iou", type=float, default=0.5)
    draw_parser.set_defaults(function=draw)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
