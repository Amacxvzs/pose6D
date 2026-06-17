from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def load_detections(split: str, detection_name: str) -> dict[tuple[str, str], dict]:
    path = PROJECT_ROOT / "outputs" / detection_name / split / "detections.jsonl"
    detections: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for det in record.get("detections", []):
                det_id = str(det["det_id"])
                detections[(record["image"], det_id)] = {
                    "image": record["image"],
                    "det_id": det_id,
                    "xyxy": [float(v) for v in det["xyxy"]],
                    "confidence": float(det["confidence"]),
                }
    return detections


def load_csv_by_key(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["image"], str(row["det_id"]))] = row
    return rows


def group_by_image(items: dict[tuple[str, str], dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in items.values():
        grouped.setdefault(item["image"], []).append(item)
    return grouped


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def evaluate_one(
    key: tuple[str, str],
    det: dict,
    dets_in_image: list[dict],
    cloud: dict,
    pose: dict,
) -> dict:
    z_near = to_float(cloud, "z_near_mm", default=0.0)
    point_count = int(to_float(cloud, "point_count", default=0.0))
    component_count = int(to_float(cloud, "component_count", default=0.0))
    selected_area = int(to_float(cloud, "selected_component_area", default=0.0))
    final_iou = to_float(pose, "final_iou", default=0.0)
    final_center_error = to_float(pose, "final_center_error_px", default=999.0)
    fitness = to_float(pose, "fitness", default=0.0)
    rmse = to_float(pose, "rmse_mm", default=999.0)
    valid = int(to_float(pose, "valid", default=0.0))

    overlap_max = 0.0
    front_occluder_count = 0
    front_occluder_max_iou = 0.0
    for other in dets_in_image:
        if other["det_id"] == det["det_id"]:
            continue
        iou = bbox_iou(det["xyxy"], other["xyxy"])
        overlap_max = max(overlap_max, iou)
        other_cloud = cloud_by_key.get((other["image"], other["det_id"]), {})
        other_z = to_float(other_cloud, "z_near_mm", default=0.0)
        if iou >= 0.03 and other_z > 0 and z_near > other_z + 8.0:
            front_occluder_count += 1
            front_occluder_max_iou = max(front_occluder_max_iou, iou)

    score = 100.0
    score -= max(0.0, 0.55 - final_iou) * 55.0
    score -= min(24.0, max(0.0, final_center_error - 12.0) * 0.8)
    score -= min(20.0, max(0.0, rmse - 5.0) * 2.5)
    score -= min(18.0, max(0.0, 0.70 - fitness) * 25.0)
    score -= min(25.0, overlap_max * 45.0)
    score -= front_occluder_count * 18.0
    if component_count > 1:
        score -= min(12.0, (component_count - 1) * 4.0)
    if point_count < 3500:
        score -= min(25.0, (3500 - point_count) / 100.0)
    if not valid:
        score -= 35.0
    score = float(np.clip(score, 0.0, 100.0))

    if front_occluder_count > 0 or overlap_max >= 0.18:
        occlusion_level = "occluded"
    elif overlap_max >= 0.05 or component_count > 1:
        occlusion_level = "partial"
    else:
        occlusion_level = "clear"

    reliable = (
        valid
        and score >= 70.0
        and front_occluder_count == 0
        and final_iou >= 0.45
        and final_center_error <= 30.0
        and point_count >= 3500
    )
    usable = valid and score >= 50.0 and final_iou >= 0.30 and final_center_error <= 55.0

    if reliable:
        quality_level = "reliable"
        action = "use_pose"
    elif usable:
        quality_level = "usable"
        action = "use_with_caution"
    else:
        quality_level = "low_confidence"
        action = "skip_or_recapture"

    if front_occluder_count > 0:
        action = "estimate_front_objects_first"

    return {
        "image": key[0],
        "det_id": key[1],
        "confidence": f"{det['confidence']:.6f}",
        "valid": valid,
        "quality_score": f"{score:.1f}",
        "quality_level": quality_level,
        "occlusion_level": occlusion_level,
        "front_occluder_count": front_occluder_count,
        "front_occluder_max_iou": f"{front_occluder_max_iou:.4f}",
        "box_overlap_max": f"{overlap_max:.4f}",
        "point_count": point_count,
        "component_count": component_count,
        "selected_component_area": selected_area,
        "z_near_mm": f"{z_near:.1f}",
        "fitness": f"{fitness:.6f}",
        "rmse_mm": f"{rmse:.3f}",
        "final_iou": f"{final_iou:.6f}",
        "final_center_error_px": f"{final_center_error:.3f}",
        "recommended_action": action,
    }


def draw_quality_vis(split: str, class_name: str, rows: list[dict], detections: dict[tuple[str, str], dict]) -> None:
    out_dir = PROJECT_ROOT / "outputs" / "pose_quality" / split / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / class_name
    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)

    colors = {
        "reliable": (0, 210, 0),
        "usable": (0, 180, 255),
        "low_confidence": (0, 0, 255),
    }
    for image_name, image_rows in by_image.items():
        img = cv2.imread(str(rgb_dir / image_name))
        if img is None:
            continue
        canvas = img.copy()
        for row in image_rows:
            det = detections.get((row["image"], row["det_id"]))
            if not det:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
            color = colors.get(row["quality_level"], (255, 255, 255))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{row['det_id']} {row['quality_level']} {row['quality_score']}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if row["occlusion_level"] != "clear":
                cv2.putText(canvas, row["occlusion_level"], (x1, min(canvas.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(out_dir / image_name), canvas)


def run(split: str, class_name: str, detection_name: str, save_vis: bool) -> None:
    global cloud_by_key
    detections = load_detections(split, detection_name)
    dets_grouped = group_by_image(detections)
    cloud_by_key = load_csv_by_key(PROJECT_ROOT / "outputs" / "object_clouds" / split / "object_clouds.csv")
    poses = load_csv_by_key(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv")

    out_dir = PROJECT_ROOT / "outputs" / "pose_quality" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for key, pose in poses.items():
        det = detections.get(key)
        cloud = cloud_by_key.get(key)
        if not det or not cloud:
            continue
        rows.append(evaluate_one(key, det, dets_grouped.get(key[0], []), cloud, pose))

    fieldnames = [
        "image",
        "det_id",
        "confidence",
        "valid",
        "quality_score",
        "quality_level",
        "occlusion_level",
        "front_occluder_count",
        "front_occluder_max_iou",
        "box_overlap_max",
        "point_count",
        "component_count",
        "selected_component_area",
        "z_near_mm",
        "fitness",
        "rmse_mm",
        "final_iou",
        "final_center_error_px",
        "recommended_action",
    ]
    with (out_dir / "pose_quality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "pose_quality.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pose_rows = load_csv_by_key(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv")
    accepted_keys = {
        (row["image"], row["det_id"])
        for row in rows
        if row["recommended_action"] in {"use_pose", "use_with_caution"}
    }
    review_keys = {
        (row["image"], row["det_id"])
        for row in rows
        if row["recommended_action"] not in {"use_pose", "use_with_caution"}
    }
    if pose_rows:
        pose_fieldnames = list(next(iter(pose_rows.values())).keys())
        quality_by_key = {(row["image"], row["det_id"]): row for row in rows}
        extra_fields = ["quality_score", "quality_level", "occlusion_level", "recommended_action"]
        with (out_dir / "accepted_poses.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pose_fieldnames + extra_fields)
            writer.writeheader()
            for key, pose_row in pose_rows.items():
                if key in accepted_keys:
                    q = quality_by_key[key]
                    writer.writerow({**pose_row, **{name: q[name] for name in extra_fields}})
        with (out_dir / "review_poses.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pose_fieldnames + extra_fields)
            writer.writeheader()
            for key, pose_row in pose_rows.items():
                if key in review_keys:
                    q = quality_by_key[key]
                    writer.writerow({**pose_row, **{name: q[name] for name in extra_fields}})
    if save_vis:
        draw_quality_vis(split, class_name, rows, detections)

    level_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        level_counts[row["quality_level"]] = level_counts.get(row["quality_level"], 0) + 1
        action_counts[row["recommended_action"]] = action_counts.get(row["recommended_action"], 0) + 1
    print(f"{split}: poses={len(rows)} accepted={len(accepted_keys)} review={len(review_keys)} quality={level_counts} actions={action_counts} -> {out_dir / 'pose_quality.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify pose reliability and occlusion risk for stacked plate parts.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--detection-name", default="detections_pose")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(split, args.class_name, args.detection_name, save_vis=not args.no_vis)


if __name__ == "__main__":
    cloud_by_key: dict[tuple[str, str], dict] = {}
    main()
