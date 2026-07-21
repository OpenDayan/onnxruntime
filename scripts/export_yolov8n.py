#!/usr/bin/env python3
import argparse
import os
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)
    exported = YOLO("yolov8n.pt").export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=13,
        nms=False,
        device="cpu",
    )
    print(Path(exported).resolve())


if __name__ == "__main__":
    main()
