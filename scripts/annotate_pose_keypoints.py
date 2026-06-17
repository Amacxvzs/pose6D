from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINT_NAMES = [
    "xmin,ymin,zmin",
    "xmax,ymin,zmin",
    "xmax,ymax,zmin",
    "xmin,ymax,zmin",
    "xmin,ymin,zmax",
    "xmax,ymin,zmax",
    "xmax,ymax,zmax",
    "xmin,ymax,zmax",
]


class PoseAnnotator:
    def __init__(self, split: str, class_name: str, detection_name: str, scale: float) -> None:
        self.split = split
        self.class_name = class_name
        self.scale = scale
        self.image_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / class_name
        self.detection_path = PROJECT_ROOT / "outputs" / detection_name / split / "detections.jsonl"
        self.output_dir = PROJECT_ROOT / "data" / "labels" / "pose_keypoints" / split / class_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.items = self.load_items()
        self.index = 0
        self.points: list[tuple[float, float]] = []
        self.window = "pose keypoint annotator"

    def load_items(self) -> list[dict]:
        items = []
        with self.detection_path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for det in record["detections"]:
                    items.append({"image": record["image"], "det": det})
        if not items:
            raise RuntimeError(f"No detections found: {self.detection_path}")
        return items

    def label_path(self, item: dict) -> Path:
        stem = Path(item["image"]).stem
        det_id = item["det"]["det_id"]
        return self.output_dir / f"{stem}_det{det_id:02d}.json"

    def load_existing(self) -> None:
        self.points = []
        path = self.label_path(self.items[self.index])
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.points = [(float(p[0]), float(p[1])) for p in data.get("points_2d", [])]

    def save(self) -> None:
        item = self.items[self.index]
        data = {
            "image": item["image"],
            "split": self.split,
            "class_name": self.class_name,
            "det_id": item["det"]["det_id"],
            "det_confidence": item["det"]["confidence"],
            "det_xyxy": item["det"]["xyxy"],
            "point_order": POINT_NAMES,
            "points_2d": [[round(x, 3), round(y, 3)] for x, y in self.points],
            "complete": len(self.points) == 8,
            "aabb_file": "models/plate/aabb_corners.json",
            "intrinsic_file": "data/calibration/camera_intrinsic_crop_720.json",
        }
        self.label_path(item).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def draw(self) -> np.ndarray:
        item = self.items[self.index]
        img = cv2.imread(str(self.image_dir / item["image"]))
        if img is None:
            raise FileNotFoundError(self.image_dir / item["image"])
        x1, y1, x2, y2 = [int(round(v)) for v in item["det"]["xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
            cv2.putText(img, str(i + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        next_name = POINT_NAMES[len(self.points)] if len(self.points) < 8 else "complete"
        text = f"{self.index + 1}/{len(self.items)} {item['image']} det={item['det']['det_id']} conf={item['det']['confidence']:.2f}"
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(img, f"next: {len(self.points)+1 if len(self.points)<8 else '-'} {next_name}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
        cv2.putText(img, "left click point | u undo | s save | n next | p prev | d skip/delete | q quit", (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        return img

    def mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= 8:
            return
        self.points.append((x / self.scale, y / self.scale))

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)
        self.load_existing()
        while True:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and self.points:
                self.points.pop()
            elif key == ord("s"):
                self.save()
            elif key == ord("n"):
                self.save()
                self.index = min(len(self.items) - 1, self.index + 1)
                self.load_existing()
            elif key == ord("p"):
                self.save()
                self.index = max(0, self.index - 1)
                self.load_existing()
            elif key == ord("d"):
                path = self.label_path(self.items[self.index])
                if path.exists():
                    path.unlink()
                self.index = min(len(self.items) - 1, self.index + 1)
                self.load_existing()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate 2D projections of 8 AABB corners for PnP.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--detection-name", default="detections_pose")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    PoseAnnotator(args.split, args.class_name, args.detection_name, args.scale).run()


if __name__ == "__main__":
    main()
