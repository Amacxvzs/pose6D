from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv9 for the plate dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="plate_yolov9t")
    parser.add_argument("--model", default="yolov9t.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(PROJECT_ROOT / "configs" / "plate_yolo.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
