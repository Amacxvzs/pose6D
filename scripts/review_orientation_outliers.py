from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pose_path = PROJECT_ROOT / "outputs" / "two_stage_poses" / "test" / "two_stage_poses.csv"
    vis_dir = PROJECT_ROOT / "outputs" / "two_stage_review" / "test" / "vis"
    output_dir = PROJECT_ROOT / "outputs" / "orientation_review"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    review: list[dict] = []
    with pose_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rvec = np.asarray([float(row["rx"]), float(row["ry"]), float(row["rz"])], dtype=np.float64).reshape(3, 1)
            rotation, _ = cv2.Rodrigues(rvec)
            normal = rotation[:, 2]
            tilt = math.degrees(math.acos(np.clip(abs(float(normal[2])), 0.0, 1.0)))
            if tilt <= 35.0:
                continue
            review.append(
                {
                    "image": row["image"],
                    "det_id": row["det_id"],
                    "normal_tilt_deg": f"{tilt:.2f}",
                    "final_iou": row["final_iou"],
                    "center_error_px": row["final_center_error_px"],
                    "source": row["two_stage_source"],
                }
            )

    copied: set[str] = set()
    for index, row in enumerate(sorted(review, key=lambda r: float(r["normal_tilt_deg"]), reverse=True), start=1):
        source = vis_dir / row["image"]
        if source.exists() and row["image"] not in copied:
            target = image_dir / f"{index:02d}_tilt{float(row['normal_tilt_deg']):.1f}_{row['image']}"
            shutil.copy2(source, target)
            copied.add(row["image"])

    with (output_dir / "orientation_outliers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "det_id", "normal_tilt_deg", "final_iou", "center_error_px", "source"],
        )
        writer.writeheader()
        writer.writerows(sorted(review, key=lambda r: float(r["normal_tilt_deg"]), reverse=True))

    print(f"orientation_outliers={len(review)} images={len(copied)}")
    print(output_dir / "orientation_outliers.csv")
    print(image_dir)


if __name__ == "__main__":
    main()
