from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_intrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    k = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64)
    return k, dist


def load_aabb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    corners = np.asarray(data["corners_3d"], dtype=np.float64)
    top_center = corners[4:8].mean(axis=0)
    return corners, top_center


def load_object_radius(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    sx = float(data["size"]["x"])
    sy = float(data["size"]["y"])
    return 0.25 * (sx + sy)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 0.999999:
        return np.eye(3, dtype=np.float64)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def stable_inplane_rotation(rmat: np.ndarray) -> np.ndarray:
    normal = rmat[:, 2]
    camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = camera_x - normal * float(np.dot(camera_x, normal))
    if np.linalg.norm(x_axis) < 1e-6:
        camera_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = camera_y - normal * float(np.dot(camera_y, normal))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, normal])


def draw_pose(image: np.ndarray, corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pts, _ = cv2.projectPoints(corners, rvec, tvec, k, dist)
    pts = pts.reshape(-1, 2).astype(int)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    out = image.copy()
    for a, b in edges:
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 255), 2)
    for i, p in enumerate(pts):
        cv2.circle(out, tuple(p), 3, (0, 0, 255), -1)
        cv2.putText(out, str(i + 1), tuple(p + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return out


def draw_axisymmetric_pose(image: np.ndarray, center: np.ndarray, normal: np.ndarray, radius_mm: float, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    circle = center.reshape(1, 3) + radius_mm * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    tip = center + normal * 45.0
    pts3 = np.vstack([circle, center.reshape(1, 3), tip.reshape(1, 3)]).astype(np.float64)
    pts2, _ = cv2.projectPoints(pts3, np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), k, dist)
    pts2 = np.rint(pts2.reshape(-1, 2)).astype(np.int32)
    out = image.copy()
    cv2.polylines(out, [pts2[: len(circle)]], isClosed=True, color=(0, 255, 0), thickness=2)
    c = tuple(int(x) for x in pts2[-2])
    t = tuple(int(x) for x in pts2[-1])
    cv2.arrowedLine(out, c, t, (0, 255, 255), 2, tipLength=0.25)
    return out


def run(split: str, class_name: str, cloud_name: str, output_name: str, save_vis: bool) -> None:
    k, dist = load_intrinsic(PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic_crop_720.json")
    aabb_path = PROJECT_ROOT / "models" / class_name / "aabb_corners.json"
    corners, top_center = load_aabb(aabb_path)
    radius_mm = load_object_radius(aabb_path)
    cloud_csv = PROJECT_ROOT / "outputs" / cloud_name / split / "object_clouds.csv"
    rgb_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / class_name
    out_dir = PROJECT_ROOT / "outputs" / output_name / split
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    poses = []
    rows = []
    by_image: dict[str, list[dict]] = {}
    with cloud_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] != "1":
                continue
            centroid = np.array([float(row["cx_mm"]), float(row["cy_mm"]), float(row["cz_mm"])], dtype=np.float64)
            normal = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])], dtype=np.float64)
            rmat = stable_inplane_rotation(rotation_between(np.array([0.0, 0.0, 1.0]), normal))
            tvec = centroid - rmat @ top_center
            rvec, _ = cv2.Rodrigues(rmat)
            pose = {
                "image": row["image"],
                "det_id": int(row["det_id"]),
                "confidence": float(row["confidence"]),
                "translation_mm": [float(v) for v in tvec],
                "rotation_matrix": [[float(v) for v in line] for line in rmat],
                "rvec": [float(v) for v in rvec.reshape(-1)],
                "normal": [float(v) for v in normal],
                "yaw_observable": False,
                "point_count": int(row["point_count"]),
                "note": "Axisymmetric initial pose from RGB-D centroid and PCA normal; yaw around disk symmetry axis is not observable.",
            }
            poses.append(pose)
            by_image.setdefault(row["image"], []).append(pose)
            rows.append(
                {
                    "image": row["image"],
                    "det_id": row["det_id"],
                    "confidence": row["confidence"],
                    "tx_mm": f"{tvec[0]:.3f}",
                    "ty_mm": f"{tvec[1]:.3f}",
                    "tz_mm": f"{tvec[2]:.3f}",
                    "rx": f"{rvec[0,0]:.6f}",
                    "ry": f"{rvec[1,0]:.6f}",
                    "rz": f"{rvec[2,0]:.6f}",
                    "normal_x": f"{normal[0]:.6f}",
                    "normal_y": f"{normal[1]:.6f}",
                    "normal_z": f"{normal[2]:.6f}",
                    "yaw_observable": 0,
                    "point_count": row["point_count"],
                }
            )

    with (out_dir / "initial_poses.jsonl").open("w", encoding="utf-8") as f:
        for pose in poses:
            f.write(json.dumps(pose, ensure_ascii=False) + "\n")
    with (out_dir / "initial_poses.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "det_id", "confidence", "tx_mm", "ty_mm", "tz_mm", "rx", "ry", "rz", "normal_x", "normal_y", "normal_z", "yaw_observable", "point_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if save_vis:
        for image_name, image_poses in by_image.items():
            img = cv2.imread(str(rgb_dir / image_name))
            if img is None:
                continue
            canvas = img
            for pose in image_poses:
                center = np.asarray(pose["translation_mm"], dtype=np.float64) + np.asarray(pose["rotation_matrix"], dtype=np.float64) @ top_center
                normal = np.asarray(pose["normal"], dtype=np.float64)
                canvas = draw_axisymmetric_pose(canvas, center, normal, radius_mm, k, dist)
            cv2.imwrite(str(vis_dir / image_name), canvas)

    print(f"{split}: poses={len(poses)} -> {out_dir / 'initial_poses.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate coarse 6D poses from extracted object point clouds.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--cloud-name", default="object_clouds")
    parser.add_argument("--output-name", default="initial_poses")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(split, args.class_name, args.cloud_name, args.output_name, save_vis=not args.no_vis)


if __name__ == "__main__":
    main()
