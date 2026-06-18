from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from refine_poses_icp import (
    bbox_iou,
    center_distance,
    load_intrinsic,
    load_object_geometry,
    projected_bbox,
    transform_from_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_full_intrinsic() -> tuple[float, float, float, float]:
    data = json.loads((PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic.json").read_text(encoding="utf-8"))
    k = data["camera_matrix"]
    return float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])


def load_detections(split: str) -> dict[str, list[dict]]:
    path = PROJECT_ROOT / "outputs" / "detections_pose" / split / "detections.jsonl"
    records: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["image"]] = record.get("detections", [])
    return records


def load_meta(split: str, class_name: str, image_name: str) -> dict:
    path = PROJECT_ROOT / "data" / "raw" / "meta" / split / class_name / f"{Path(image_name).stem}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rgb_crop_x": 0, "rgb_crop_y": 0}


def fit_table_plane(
    depth: np.ndarray,
    full_size: tuple[int, int],
    detections: list[dict],
    crop_x: float,
    crop_y: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    exclusion_margin_px: int,
) -> tuple[np.ndarray, float, float, int]:
    width, height = full_size
    depth_full = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = (depth_full >= 250) & (depth_full <= 2500)
    valid[: int(height * 0.08), :] = False
    valid[int(height * 0.94) :, :] = False
    valid[:, : int(width * 0.03)] = False
    valid[:, int(width * 0.97) :] = False

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        x1 = max(0, int(round(x1 + crop_x)) - exclusion_margin_px)
        y1 = max(0, int(round(y1 + crop_y)) - exclusion_margin_px)
        x2 = min(width - 1, int(round(x2 + crop_x)) + exclusion_margin_px)
        y2 = min(height - 1, int(round(y2 + crop_y)) + exclusion_margin_px)
        valid[y1 : y2 + 1, x1 : x2 + 1] = False

    ys, xs = np.where(valid)
    if len(xs) < 1000:
        raise RuntimeError("Not enough valid table depth pixels")
    stride = max(1, len(xs) // 50000)
    xs = xs[::stride].astype(np.float64)
    ys = ys[::stride].astype(np.float64)
    z = depth_full[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.column_stack([x, y, z])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    plane, inliers = cloud.segment_plane(distance_threshold=5.0, ransac_n=3, num_iterations=800)
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] > 0:
        normal = -normal
        plane = [-v for v in plane]
    inlier_ratio = len(inliers) / len(points)
    residual = np.abs(points @ normal + float(plane[3]))
    rmse = float(np.sqrt(np.mean(np.square(residual[inliers])))) if inliers else float("inf")
    return normal, inlier_ratio, rmse, len(points)


def clamp_normal(current: np.ndarray, reference: np.ndarray, max_angle_deg: float) -> tuple[np.ndarray, float, bool]:
    current = current / np.linalg.norm(current)
    reference = reference / np.linalg.norm(reference)
    if float(np.dot(current, reference)) < 0:
        current = -current
    angle = math.degrees(math.acos(np.clip(float(np.dot(current, reference)), -1.0, 1.0)))
    if angle <= max_angle_deg:
        return current, angle, False
    fraction = max_angle_deg / angle
    theta = math.radians(angle)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-8:
        corrected = reference
    else:
        corrected = (
            math.sin((1.0 - fraction) * theta) / sin_theta * reference
            + math.sin(fraction * theta) / sin_theta * current
        )
    corrected /= np.linalg.norm(corrected)
    return corrected, angle, True


def rotation_with_normal(current_rotation: np.ndarray, normal: np.ndarray) -> np.ndarray:
    x_axis = current_rotation[:, 0] - normal * float(np.dot(current_rotation[:, 0], normal))
    if np.linalg.norm(x_axis) < 1e-6:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = helper - normal * float(np.dot(helper, normal))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, normal])


def run(split: str, class_name: str, max_angle_deg: float) -> dict:
    input_rows = read_csv(PROJECT_ROOT / "outputs" / "two_stage_poses" / split / "two_stage_poses.csv")
    detections_by_image = load_detections(split)
    detection_boxes = {
        (image, str(det["det_id"])): [float(v) for v in det["xyxy"]]
        for image, detections in detections_by_image.items()
        for det in detections
    }
    rgb_full_dir = PROJECT_ROOT / "data" / "raw" / "rgb_full" / split / class_name
    depth_dir = PROJECT_ROOT / "data" / "raw" / "depth" / split / class_name
    fx, fy, cx, cy = load_full_intrinsic()
    k_crop, dist_crop = load_intrinsic(PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic_crop_720.json")
    corners, _radius, top_center = load_object_geometry(PROJECT_ROOT / "models" / class_name / "aabb_corners.json")

    planes: dict[str, dict] = {}
    output_rows: list[dict] = []
    for row in input_rows:
        image_name = row["image"]
        if image_name not in planes:
            rgb_full = cv2.imread(str(rgb_full_dir / image_name))
            depth = cv2.imread(str(depth_dir / image_name), cv2.IMREAD_UNCHANGED)
            meta = load_meta(split, class_name, image_name)
            normal, ratio, rmse, samples = fit_table_plane(
                depth,
                (rgb_full.shape[1], rgb_full.shape[0]),
                detections_by_image.get(image_name, []),
                float(meta.get("rgb_crop_x", 0)),
                float(meta.get("rgb_crop_y", 0)),
                fx,
                fy,
                cx,
                cy,
                exclusion_margin_px=35,
            )
            planes[image_name] = {
                "normal": normal,
                "inlier_ratio": ratio,
                "rmse_mm": rmse,
                "samples": samples,
            }

        plane = planes[image_name]
        transform = transform_from_pose(row)
        current_normal = transform[:3, 2]
        corrected_normal, original_angle, constrained = clamp_normal(current_normal, plane["normal"], max_angle_deg)
        center = transform[:3, :3] @ top_center + transform[:3, 3]
        corrected_rotation = rotation_with_normal(transform[:3, :3], corrected_normal)
        corrected_translation = center - corrected_rotation @ top_center
        corrected = np.eye(4, dtype=np.float64)
        corrected[:3, :3] = corrected_rotation
        corrected[:3, 3] = corrected_translation

        det_box = detection_boxes.get((image_name, str(row["det_id"])))
        corrected_box = projected_bbox(corners, corrected, k_crop, dist_crop)
        corrected_iou = bbox_iou(corrected_box, det_box) if corrected_box is not None and det_box is not None else 0.0
        corrected_center_error = center_distance(corrected_box, det_box)
        plane_pose_valid = (
            str(row.get("two_stage_valid", "0")) == "1"
            and corrected_iou >= 0.45
            and corrected_center_error <= 30.0
        )
        rvec, _ = cv2.Rodrigues(corrected_rotation)
        out = dict(row)
        out.update(
            {
                "tx_mm": f"{corrected_translation[0]:.3f}",
                "ty_mm": f"{corrected_translation[1]:.3f}",
                "tz_mm": f"{corrected_translation[2]:.3f}",
                "rx": f"{rvec[0, 0]:.6f}",
                "ry": f"{rvec[1, 0]:.6f}",
                "rz": f"{rvec[2, 0]:.6f}",
                "final_iou": f"{corrected_iou:.6f}",
                "final_center_error_px": f"{corrected_center_error:.3f}",
                "table_nx": f"{plane['normal'][0]:.6f}",
                "table_ny": f"{plane['normal'][1]:.6f}",
                "table_nz": f"{plane['normal'][2]:.6f}",
                "table_inlier_ratio": f"{plane['inlier_ratio']:.6f}",
                "table_rmse_mm": f"{plane['rmse_mm']:.3f}",
                "normal_angle_before_deg": f"{original_angle:.3f}",
                "normal_constrained": int(constrained),
                "normal_angle_after_deg": f"{min(original_angle, max_angle_deg):.3f}",
                "plane_pose_valid": int(plane_pose_valid),
            }
        )
        output_rows.append(out)

    out_dir = PROJECT_ROOT / "outputs" / "two_stage_poses_table_constrained" / split
    extra_fields = [
        "table_nx",
        "table_ny",
        "table_nz",
        "table_inlier_ratio",
        "table_rmse_mm",
        "normal_angle_before_deg",
        "normal_constrained",
        "normal_angle_after_deg",
        "plane_pose_valid",
    ]
    write_csv(out_dir / "two_stage_poses.csv", output_rows, list(input_rows[0].keys()) + extra_fields)
    plane_rows = [
        {
            "image": image,
            "nx": f"{data['normal'][0]:.6f}",
            "ny": f"{data['normal'][1]:.6f}",
            "nz": f"{data['normal'][2]:.6f}",
            "inlier_ratio": f"{data['inlier_ratio']:.6f}",
            "rmse_mm": f"{data['rmse_mm']:.3f}",
            "samples": data["samples"],
        }
        for image, data in sorted(planes.items())
    ]
    write_csv(out_dir / "table_planes.csv", plane_rows, ["image", "nx", "ny", "nz", "inlier_ratio", "rmse_mm", "samples"])
    constrained_count = sum(int(row["normal_constrained"]) for row in output_rows)
    valid_count = sum(int(row["plane_pose_valid"]) for row in output_rows)
    print(f"{split}: poses={len(output_rows)} constrained={constrained_count} valid={valid_count} planes={len(planes)}")
    print(out_dir / "two_stage_poses.csv")
    print(out_dir / "table_planes.csv")
    total = len(output_rows) + len(
        read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "skip_or_recapture.csv")
    )
    return {
        "split": split,
        "total": total,
        "position_valid": len(output_rows),
        "normal_constrained": constrained_count,
        "strict_valid": valid_count,
        "strict_valid_rate": valid_count / total if total else 0.0,
        "rejected_after_constraint": len(output_rows) - valid_count,
        "skip": total - len(output_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrain plate pose normals to the fitted table plane.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--max-angle-deg", type=float, default=20.0)
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    summaries = [run(split, args.class_name, args.max_angle_deg) for split in splits]
    summary_path = PROJECT_ROOT / "outputs" / "two_stage_poses_table_constrained" / "summary.md"
    lines = [
        "# Table-Plane-Constrained Pose Summary",
        "",
        f"Maximum normal deviation from the fitted table plane: {args.max_angle_deg:.1f} deg",
        "",
        "| split | total | position valid | normals corrected | strict valid | strict rate | rejected after constraint | skip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['total']} | {row['position_valid']} | "
            f"{row['normal_constrained']} | {row['strict_valid']} | "
            f"{row['strict_valid_rate'] * 100:.1f}% | {row['rejected_after_constraint']} | {row['skip']} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
