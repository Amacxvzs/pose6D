from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from refine_poses_icp import draw_axisymmetric_pose, load_intrinsic, load_object_geometry, transform_from_pose


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_detections(split: str) -> dict[tuple[str, str], dict]:
    path = PROJECT_ROOT / "outputs" / "detections_pose" / split / "detections.jsonl"
    detections: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for det in record.get("detections", []):
                detections[(record["image"], str(det["det_id"]))] = det
    return detections


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def make_contact_sheets(vis_dir: Path, output_dir: Path, cols: int = 4, rows: int = 4) -> list[Path]:
    files = sorted(vis_dir.glob("*.png"))
    per_page = cols * rows
    paths: list[Path] = []
    for page_index, start in enumerate(range(0, len(files), per_page), start=1):
        panels: list[np.ndarray] = []
        for path in files[start : start + per_page]:
            image = cv2.imread(str(path))
            if image is None:
                continue
            panel = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
            draw_label(panel, path.stem[-18:], (8, 22), (255, 255, 255))
            panels.append(panel)
        if not panels:
            continue
        blank = np.zeros_like(panels[0])
        while len(panels) < per_page:
            panels.append(blank.copy())
        sheet_rows = [np.hstack(panels[i : i + cols]) for i in range(0, per_page, cols)]
        output = output_dir / f"two_stage_contact_sheet_{page_index:02d}.png"
        cv2.imwrite(str(output), np.vstack(sheet_rows))
        paths.append(output)
    return paths


def run(split: str, pose_name: str, output_name: str) -> None:
    pose_rows = read_csv(PROJECT_ROOT / "outputs" / pose_name / split / "two_stage_poses.csv")
    skip_rows = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "skip_or_recapture.csv")
    detections = load_detections(split)
    _, radius_mm, top_center = load_object_geometry(PROJECT_ROOT / "models" / "plate" / "aabb_corners.json")
    k, dist = load_intrinsic(PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic_crop_720.json")
    rgb_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / "plate"
    output_dir = PROJECT_ROOT / "outputs" / output_name / split
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    poses_by_image: dict[str, list[dict]] = {}
    for row in pose_rows:
        poses_by_image.setdefault(row["image"], []).append(row)
    skips_by_image: dict[str, list[dict]] = {}
    for row in skip_rows:
        skips_by_image.setdefault(row["image"], []).append(row)

    image_names = sorted(set(poses_by_image) | set(skips_by_image))
    review_rows: list[dict] = []
    for image_name in image_names:
        image = cv2.imread(str(rgb_dir / image_name))
        if image is None:
            continue
        canvas = image.copy()
        stage1_count = 0
        stage2_count = 0
        skip_count = 0

        for row in poses_by_image.get(image_name, []):
            source = row["two_stage_source"]
            color = (0, 255, 0) if source == "stage1_front_or_clear" else (255, 255, 0)
            valid_field = row.get("plane_pose_valid", row.get("two_stage_valid", "0"))
            if str(valid_field) != "1":
                color = (0, 0, 255)
            stage1_count += int(source == "stage1_front_or_clear")
            stage2_count += int(source == "stage2_after_front_mask")
            transform = transform_from_pose(row)
            canvas = draw_axisymmetric_pose(canvas, transform, top_center, radius_mm, k, dist, color)
            det = detections.get((image_name, str(row["det_id"])))
            if det:
                x1, y1, _, _ = [int(round(v)) for v in det["xyxy"]]
                label = f"id{row['det_id']} {'S1' if source == 'stage1_front_or_clear' else 'S2'} IoU={float(row['final_iou']):.2f}"
                draw_label(canvas, label, (x1, max(18, y1 - 6)), color)

        for row in skips_by_image.get(image_name, []):
            det = detections.get((image_name, str(row["det_id"])))
            if not det:
                continue
            skip_count += 1
            x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
            draw_label(canvas, f"id{row['det_id']} SKIP", (x1, max(18, y1 - 6)), (0, 0, 255))

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        header = f"S1={stage1_count} S2={stage2_count} SKIP={skip_count}"
        draw_label(canvas, header, (8, 24), (255, 255, 255))
        cv2.imwrite(str(vis_dir / image_name), canvas)
        review_rows.append(
            {
                "image": image_name,
                "stage1": stage1_count,
                "stage2": stage2_count,
                "skip": skip_count,
                "pose_total": stage1_count + stage2_count,
            }
        )

    with (output_dir / "review_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "stage1", "stage2", "skip", "pose_total"])
        writer.writeheader()
        writer.writerows(review_rows)

    sheets = make_contact_sheets(vis_dir, output_dir)
    print(f"{split}: images={len(review_rows)} poses={len(pose_rows)} skip={len(skip_rows)}")
    print(output_dir / "review_summary.csv")
    for sheet in sheets:
        print(sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize final two-stage pose outputs and skipped detections.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--pose-name", default="two_stage_poses")
    parser.add_argument("--output-name", default="two_stage_review")
    args = parser.parse_args()
    run(args.split, args.pose_name, args.output_name)


if __name__ == "__main__":
    main()
