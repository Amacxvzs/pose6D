from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def draw_box(image, box, conf: float, name: str):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{name} {conf:.2f}"
    cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def box_passes_filter(xyxy: list[float], width: int, height: int, min_size: float, max_size: float, min_aspect: float, max_aspect: float, border: float) -> bool:
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    aspect = bw / bh
    if bw < min_size or bh < min_size or bw > max_size or bh > max_size:
        return False
    if aspect < min_aspect or aspect > max_aspect:
        return False
    if x1 < border or y1 < border or x2 > width - border or y2 > height - border:
        return False
    return True


def run(split: str, class_name: str, conf: float, iou: float, imgsz: int, save_vis: bool, output_name: str, min_size: float, max_size: float, min_aspect: float, max_aspect: float, border: float) -> None:
    model_path = PROJECT_ROOT / "runs" / "detect" / "plate_yolov9t" / "weights" / "best.pt"
    image_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / class_name
    output_dir = PROJECT_ROOT / "outputs" / output_name / split
    vis_dir = output_dir / "vis"
    jsonl_path = output_dir / "detections.jsonl"
    csv_path = output_dir / "detections.csv"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    images = sorted(image_dir.glob("*.png"))
    rows = []
    json_records = []

    for image_input in images:
        results = model.predict(source=str(image_input), imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False)
        result = results[0]
        image_path = Path(result.path)
        image = cv2.imread(str(image_path)) if save_vis else None
        detections = []
        height, width = int(result.orig_shape[0]), int(result.orig_shape[1])
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                if not box_passes_filter(xyxy, width, height, min_size, max_size, min_aspect, max_aspect, border):
                    continue
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                det = {
                    "det_id": i,
                    "class_id": cls_id,
                    "class_name": model.names.get(cls_id, str(cls_id)),
                    "confidence": score,
                    "xyxy": xyxy,
                    "xywh": [
                        (xyxy[0] + xyxy[2]) / 2.0,
                        (xyxy[1] + xyxy[3]) / 2.0,
                        xyxy[2] - xyxy[0],
                        xyxy[3] - xyxy[1],
                    ],
                }
                detections.append(det)
                rows.append(
                    {
                        "image": image_path.name,
                        "det_id": i,
                        "class_id": cls_id,
                        "class_name": det["class_name"],
                        "confidence": f"{score:.6f}",
                        "x1": f"{xyxy[0]:.3f}",
                        "y1": f"{xyxy[1]:.3f}",
                        "x2": f"{xyxy[2]:.3f}",
                        "y2": f"{xyxy[3]:.3f}",
                        "xc": f"{det['xywh'][0]:.3f}",
                        "yc": f"{det['xywh'][1]:.3f}",
                        "w": f"{det['xywh'][2]:.3f}",
                        "h": f"{det['xywh'][3]:.3f}",
                    }
                )
                if image is not None:
                    draw_box(image, xyxy, score, det["class_name"])

        json_records.append(
            {
                "image": image_path.name,
                "split": split,
                "width": width,
                "height": height,
                "detections": detections,
            }
        )
        if image is not None:
            cv2.imwrite(str(vis_dir / image_path.name), image)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in json_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "det_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "xc", "yc", "w", "h"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    image_count = len(json_records)
    det_count = sum(len(r["detections"]) for r in json_records)
    empty_count = sum(1 for r in json_records if not r["detections"])
    print(f"{split}: images={image_count} detections={det_count} empty_images={empty_count}")
    print(jsonl_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained YOLOv9 detector and save reusable detection outputs.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-name", default="detections")
    parser.add_argument("--min-size", type=float, default=30.0)
    parser.add_argument("--max-size", type=float, default=170.0)
    parser.add_argument("--min-aspect", type=float, default=0.55)
    parser.add_argument("--max-aspect", type=float, default=1.8)
    parser.add_argument("--border", type=float, default=3.0)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(
            split,
            args.class_name,
            args.conf,
            args.iou,
            args.imgsz,
            save_vis=not args.no_vis,
            output_name=args.output_name,
            min_size=args.min_size,
            max_size=args.max_size,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            border=args.border,
        )


if __name__ == "__main__":
    main()
