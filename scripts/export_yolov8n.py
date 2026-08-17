#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

from ultralytics import YOLO, __version__ as ultralytics_version


EXPECTED_ULTRALYTICS_VERSION = "8.4.92"


def main() -> None:
    if ultralytics_version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            "YOLOv8n export requires ultralytics=="
            f"{EXPECTED_ULTRALYTICS_VERSION}, found {ultralytics_version}"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    model = YOLO("yolov8n.pt")
    with tempfile.TemporaryDirectory(dir=output_dir, prefix="export-") as temporary:
        os.chdir(temporary)
        exported = model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=13,
            nms=True,
            conf=0.25,
            iou=0.7,
            max_det=300,
            agnostic_nms=False,
            device="cpu",
        )
        target = output_dir / "yolov8n.onnx"
        Path(exported).replace(target)
    print(target)


if __name__ == "__main__":
    main()
