from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_intrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["camera_matrix"], dtype=np.float64), np.asarray(data.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64)


def load_aabb(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["corners_3d"], dtype=np.float64)


def load_object_geometry(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    corners = np.asarray(data["corners_3d"], dtype=np.float64)
    sx = float(data["size"]["x"])
    sy = float(data["size"]["y"])
    radius = 0.25 * (sx + sy)
    top_center = corners[4:8].mean(axis=0)
    return corners, radius, top_center


def load_detection_boxes(split: str) -> dict[tuple[str, str], list[float]]:
    path = PROJECT_ROOT / "outputs" / "detections_pose" / split / "detections.jsonl"
    boxes: dict[tuple[str, str], list[float]] = {}
    if not path.exists():
        return boxes
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for det in record.get("detections", []):
                boxes[(record["image"], str(det["det_id"]))] = [float(v) for v in det["xyxy"]]
    return boxes


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
    return 0.0 if denom <= 0 else inter / denom


def projected_bbox(corners: np.ndarray, tform: np.ndarray, k: np.ndarray, dist: np.ndarray) -> list[float] | None:
    rvec, _ = cv2.Rodrigues(tform[:3, :3])
    tvec = tform[:3, 3].reshape(3, 1)
    pts, _ = cv2.projectPoints(corners, rvec, tvec, k, dist)
    pts = pts.reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return None
    if np.max(np.abs(pts)) > 100000:
        return None
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def center_distance(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    return float(np.hypot(acx - bcx, acy - bcy))


def load_model_cloud(model_path: Path, samples: int, voxel_mm: float, surface: str) -> o3d.geometry.PointCloud:
    mesh = trimesh.load_mesh(str(model_path), force="mesh")
    face_weight = None
    if surface == "top":
        face_weight = np.where(mesh.face_normals[:, 2] > 0.25, mesh.area_faces, 0.0)
    elif surface == "top_side":
        face_weight = np.where(mesh.face_normals[:, 2] > -0.15, mesh.area_faces, 0.0)
    elif surface != "full":
        raise ValueError(f"Unknown surface mode: {surface}")
    if face_weight is not None and float(face_weight.sum()) <= 0:
        face_weight = None
    pts, _ = trimesh.sample.sample_surface(mesh, samples, face_weight=face_weight)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if voxel_mm > 0:
        pcd = pcd.voxel_down_sample(voxel_mm)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_mm * 3.0, max_nn=30))
    return pcd


def read_ply(path: Path, voxel_mm: float) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel_mm > 0:
        pcd = pcd.voxel_down_sample(voxel_mm)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_mm * 3.0, max_nn=30))
    return pcd


def transform_from_pose(row: dict) -> np.ndarray:
    rvec = np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])], dtype=np.float64).reshape(3, 1)
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.array([float(row["tx_mm"]), float(row["ty_mm"]), float(row["tz_mm"])], dtype=np.float64)
    tform = np.eye(4, dtype=np.float64)
    tform[:3, :3] = rmat
    tform[:3, 3] = t
    return tform


def draw_pose(image: np.ndarray, corners: np.ndarray, tform: np.ndarray, k: np.ndarray, dist: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(tform[:3, :3])
    tvec = tform[:3, 3].reshape(3, 1)
    pts, _ = cv2.projectPoints(corners, rvec, tvec, k, dist)
    pts = pts.reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return image.copy()
    if np.max(np.abs(pts)) > 100000:
        return image.copy()
    pts = np.rint(pts).astype(np.int32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    out = image.copy()
    for a, b in edges:
        pt1 = (int(pts[a, 0]), int(pts[a, 1]))
        pt2 = (int(pts[b, 0]), int(pts[b, 1]))
        cv2.line(out, pt1, pt2, color, 2)
    return out


def draw_axisymmetric_pose(image: np.ndarray, tform: np.ndarray, top_center: np.ndarray, radius_mm: float, k: np.ndarray, dist: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    center = tform[:3, :3] @ top_center + tform[:3, 3]
    normal = tform[:3, 2]
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
    pts2 = pts2.reshape(-1, 2)
    if not np.all(np.isfinite(pts2)) or np.max(np.abs(pts2)) > 100000:
        return image.copy()
    pts2 = np.rint(pts2).astype(np.int32)
    out = image.copy()
    cv2.polylines(out, [pts2[: len(circle)]], isClosed=True, color=color, thickness=2)
    cv2.arrowedLine(out, tuple(pts2[-2]), tuple(pts2[-1]), color, 2, tipLength=0.25)
    return out


def run(
    split: str,
    class_name: str,
    init_name: str,
    cloud_name: str,
    output_name: str,
    voxel_mm: float,
    max_corr_mm: float,
    icp_iters: int,
    surface: str,
    save_vis: bool,
) -> None:
    model_pcd = load_model_cloud(PROJECT_ROOT / "models" / class_name / "plate.stl", samples=12000, voxel_mm=voxel_mm, surface=surface)
    corners, radius_mm, top_center = load_object_geometry(PROJECT_ROOT / "models" / class_name / "aabb_corners.json")
    k, dist = load_intrinsic(PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic_crop_720.json")

    init_csv = PROJECT_ROOT / "outputs" / init_name / split / "initial_poses.csv"
    cloud_csv = PROJECT_ROOT / "outputs" / cloud_name / split / "object_clouds.csv"
    det_boxes = load_detection_boxes(split)
    rgb_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / class_name
    out_dir = PROJECT_ROOT / "outputs" / output_name / split
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    clouds = {}
    with cloud_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clouds[(row["image"], row["det_id"])] = row

    rows = []
    records = []
    image_tforms: dict[str, list[tuple[np.ndarray, np.ndarray, dict]]] = {}
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iters)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    with init_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["image"], row["det_id"])
            cloud_row = clouds.get(key)
            if not cloud_row or not cloud_row.get("ply"):
                continue
            target = read_ply(PROJECT_ROOT / cloud_row["ply"], voxel_mm=voxel_mm)
            init = transform_from_pose(row)
            result = o3d.pipelines.registration.registration_icp(
                model_pcd,
                target,
                max_corr_mm,
                init,
                estimation,
                criteria,
            )
            refined = result.transformation
            init_t = init[:3, 3]
            ref_t = refined[:3, 3]
            delta_t = float(np.linalg.norm(ref_t - init_t))
            det_box = det_boxes.get((row["image"], row["det_id"]))
            init_box = projected_bbox(corners, init, k, dist)
            init_iou = bbox_iou(init_box, det_box) if init_box is not None and det_box is not None else 0.0
            init_center_error = center_distance(init_box, det_box)
            proj_box = projected_bbox(corners, refined, k, dist)
            proj_iou = bbox_iou(proj_box, det_box) if proj_box is not None and det_box is not None else 0.0
            proj_center_error = center_distance(proj_box, det_box)
            icp_valid = (
                float(result.fitness) >= 0.6
                and float(result.inlier_rmse) <= 15.0
                and delta_t <= 120.0
                and 250.0 <= float(ref_t[2]) <= 1500.0
                and proj_iou >= 0.20
                and proj_center_error <= 55.0
            )
            init_valid = (
                250.0 <= float(init_t[2]) <= 1500.0
                and init_iou >= 0.20
                and init_center_error <= 55.0
            )
            use_icp = (
                icp_valid
                and proj_iou >= init_iou - 0.05
                and proj_center_error <= init_center_error + 10.0
            )
            final = refined if use_icp else init
            final_t = final[:3, 3]
            final_box = proj_box if use_icp else init_box
            final_iou = proj_iou if use_icp else init_iou
            final_center_error = proj_center_error if use_icp else init_center_error
            valid = (icp_valid if use_icp else init_valid)
            rvec, _ = cv2.Rodrigues(final[:3, :3])
            rec = {
                "image": row["image"],
                "det_id": int(row["det_id"]),
                "confidence": float(row["confidence"]),
                "valid": bool(valid),
                "source": "icp" if use_icp else "initial",
                "init_iou": float(init_iou),
                "init_center_error_px": float(init_center_error),
                "fitness": float(result.fitness),
                "rmse_mm": float(result.inlier_rmse),
                "delta_t_mm": delta_t,
                "proj_iou": float(proj_iou),
                "proj_center_error_px": float(proj_center_error),
                "final_iou": float(final_iou),
                "final_center_error_px": float(final_center_error),
                "translation_mm": [float(v) for v in final_t],
                "rvec": [float(v) for v in rvec.reshape(-1)],
                "rotation_matrix": [[float(v) for v in line] for line in final[:3, :3]],
                "point_count": int(cloud_row["point_count"]),
            }
            records.append(rec)
            rows.append(
                {
                    "image": rec["image"],
                    "det_id": rec["det_id"],
                    "confidence": f"{rec['confidence']:.6f}",
                    "valid": int(rec["valid"]),
                    "source": rec["source"],
                    "init_iou": f"{rec['init_iou']:.6f}",
                    "init_center_error_px": f"{rec['init_center_error_px']:.3f}",
                    "fitness": f"{rec['fitness']:.6f}",
                    "rmse_mm": f"{rec['rmse_mm']:.3f}",
                    "delta_t_mm": f"{rec['delta_t_mm']:.3f}",
                    "proj_iou": f"{rec['proj_iou']:.6f}",
                    "proj_center_error_px": f"{rec['proj_center_error_px']:.3f}",
                    "final_iou": f"{rec['final_iou']:.6f}",
                    "final_center_error_px": f"{rec['final_center_error_px']:.3f}",
                    "tx_mm": f"{final_t[0]:.3f}",
                    "ty_mm": f"{final_t[1]:.3f}",
                    "tz_mm": f"{final_t[2]:.3f}",
                    "rx": f"{rvec[0,0]:.6f}",
                    "ry": f"{rvec[1,0]:.6f}",
                    "rz": f"{rvec[2,0]:.6f}",
                    "point_count": rec["point_count"],
                }
            )
            image_tforms.setdefault(rec["image"], []).append((init, final, rec))

    with (out_dir / "icp_poses.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "icp_poses.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "det_id", "confidence", "valid", "source", "init_iou", "init_center_error_px", "fitness", "rmse_mm", "delta_t_mm", "proj_iou", "proj_center_error_px", "final_iou", "final_center_error_px", "tx_mm", "ty_mm", "tz_mm", "rx", "ry", "rz", "point_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if save_vis:
        for image_name, transforms in image_tforms.items():
            img = cv2.imread(str(rgb_dir / image_name))
            if img is None:
                continue
            canvas = img.copy()
            for init, refined, rec in transforms:
                canvas = draw_axisymmetric_pose(canvas, init, top_center, radius_mm, k, dist, (0, 255, 255))
                color = (0, 255, 0) if rec["valid"] else (0, 0, 255)
                canvas = draw_axisymmetric_pose(canvas, refined, top_center, radius_mm, k, dist, color)
            cv2.imwrite(str(vis_dir / image_name), canvas)

    if rows:
        rmse = [float(r["rmse_mm"]) for r in rows]
        fitness = [float(r["fitness"]) for r in rows]
        valid_count = sum(int(r["valid"]) for r in rows)
        print(f"{split}: poses={len(rows)} valid={valid_count} fitness_mean={np.mean(fitness):.3f} rmse_mean={np.mean(rmse):.2f} -> {out_dir / 'icp_poses.csv'}")
    else:
        print(f"{split}: no poses")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine RGB-D initial poses using CAD-to-depth ICP.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--init-name", default="initial_poses")
    parser.add_argument("--cloud-name", default="object_clouds")
    parser.add_argument("--output-name", default="icp_poses")
    parser.add_argument("--voxel-mm", type=float, default=3.0)
    parser.add_argument("--max-corr-mm", type=float, default=25.0)
    parser.add_argument("--icp-iters", type=int, default=40)
    parser.add_argument("--surface", choices=("full", "top", "top_side"), default="top")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(
            split,
            args.class_name,
            args.init_name,
            args.cloud_name,
            args.output_name,
            args.voxel_mm,
            args.max_corr_mm,
            args.icp_iters,
            args.surface,
            save_vis=not args.no_vis,
        )


if __name__ == "__main__":
    main()
