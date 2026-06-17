from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_intrinsic(path: Path) -> tuple[float, float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    k = data["camera_matrix"]
    return float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None and len(colors) == len(points)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def pca_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)
    normal = vectors[:, order[0]]
    if normal[2] > 0:
        normal = -normal
    return normal, values[order]


def read_detections(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_meta(path: Path) -> dict:
    if not path.exists():
        return {
            "rgb_crop_x": 0,
            "rgb_crop_y": 0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def select_foreground_component(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8 > 0, {"component_count": 0, "selected_component_area": int(np.count_nonzero(mask_u8))}

    h, w = mask.shape[:2]
    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    best_label = 0
    best_score = -1e18
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        centroid = np.asarray(centroids[label], dtype=np.float32)
        distance = float(np.linalg.norm(centroid - center))
        bbox_w = float(stats[label, cv2.CC_STAT_WIDTH])
        bbox_h = float(stats[label, cv2.CC_STAT_HEIGHT])
        compactness = min(bbox_w, bbox_h) / max(bbox_w, bbox_h, 1.0)
        score = area * (0.65 + 0.35 * compactness) - 7.5 * distance
        if score > best_score:
            best_score = score
            best_label = label
            best_area = area

    if best_label == 0:
        return mask_u8 > 0, {"component_count": int(num_labels - 1), "selected_component_area": int(np.count_nonzero(mask_u8))}
    selected = labels == best_label
    selected = cv2.morphologyEx(selected.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    return selected, {"component_count": int(num_labels - 1), "selected_component_area": int(best_area)}


def extract_one(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    xyxy: list[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_min_mm: float,
    depth_max_mm: float,
    depth_window_mm: float,
    padding: int,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    h, w = rgb.shape[:2]
    depth_resized = cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = [
        int(round(xyxy[0] + crop_x)),
        int(round(xyxy[1] + crop_y)),
        int(round(xyxy[2] + crop_x)),
        int(round(xyxy[3] + crop_y)),
    ]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w - 1, x2 + padding)
    y2 = min(h - 1, y2 + padding)

    roi = depth_resized[y1 : y2 + 1, x1 : x2 + 1]
    valid = (roi >= depth_min_mm) & (roi <= depth_max_mm)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), np.zeros_like(roi, dtype=np.uint8), {"reason": "no_valid_depth"}

    valid_depths = roi[valid].astype(np.float32)
    z_near = float(np.percentile(valid_depths, 3))
    z_far = z_near + depth_window_mm
    mask = valid & (roi.astype(np.float32) <= z_far)
    if np.count_nonzero(mask) < 20:
        z_far = float(np.percentile(valid_depths, 35))
        mask = valid & (roi.astype(np.float32) <= z_far)
    mask, component_stats = select_foreground_component(mask)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), mask.astype(np.uint8), {"reason": "empty_foreground"}

    u = xs.astype(np.float32) + x1
    v = ys.astype(np.float32) + y1
    z = depth_resized[v.astype(np.int32), u.astype(np.int32)].astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = rgb[v.astype(np.int32), u.astype(np.int32), ::-1].copy()
    stats = {
        "z_near_mm": z_near,
        "z_far_mm": float(z_far),
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "foreground_pixels": int(len(points)),
        "roi_x1_full": int(x1),
        "roi_y1_full": int(y1),
        "roi_x2_full": int(x2),
        "roi_y2_full": int(y2),
        **component_stats,
    }
    return points, colors, mask.astype(np.uint8), stats


def run(split: str, detection_name: str, class_name: str, depth_window_mm: float, min_points: int, save_vis: bool) -> None:
    fx, fy, cx, cy = load_intrinsic(PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic.json")
    det_path = PROJECT_ROOT / "outputs" / detection_name / split / "detections.jsonl"
    rgb_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / split / class_name
    rgb_full_dir = PROJECT_ROOT / "data" / "raw" / "rgb_full" / split / class_name
    depth_dir = PROJECT_ROOT / "data" / "raw" / "depth" / split / class_name
    meta_dir = PROJECT_ROOT / "data" / "raw" / "meta" / split / class_name
    out_dir = PROJECT_ROOT / "outputs" / "object_clouds" / split
    ply_dir = out_dir / "ply"
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    json_records = []
    for record in read_detections(det_path):
        image_name = record["image"]
        stem = Path(image_name).stem
        rgb = cv2.imread(str(rgb_dir / image_name), cv2.IMREAD_COLOR)
        rgb_full_path = rgb_full_dir / image_name
        rgb_full = cv2.imread(str(rgb_full_path), cv2.IMREAD_COLOR)
        if rgb_full is None:
            rgb_full = rgb
        depth = cv2.imread(str(depth_dir / image_name), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        meta = load_meta(meta_dir / f"{stem}.json")
        crop_x = float(meta.get("rgb_crop_x", 0.0))
        crop_y = float(meta.get("rgb_crop_y", 0.0))
        overlay = rgb.copy()
        image_objects = []
        for det in record["detections"]:
            det_id = int(det["det_id"])
            points, colors, mask, stats = extract_one(
                rgb_full,
                depth,
                det["xyxy"],
                fx,
                fy,
                cx,
                cy,
                depth_min_mm=250,
                depth_max_mm=2500,
                depth_window_mm=depth_window_mm,
                padding=4,
                crop_x=crop_x,
                crop_y=crop_y,
            )
            ok = len(points) >= min_points
            ply_name = f"{stem}_det{det_id:02d}.ply"
            centroid = [None, None, None]
            normal = [None, None, None]
            eigenvalues = [None, None, None]
            if ok:
                normal_arr, eig = pca_normal(points)
                centroid_arr = points.mean(axis=0)
                centroid = [float(v) for v in centroid_arr]
                normal = [float(v) for v in normal_arr]
                eigenvalues = [float(v) for v in eig]
                write_ply(ply_dir / ply_name, points, colors)

            x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            if mask.size:
                x1p = max(0, int(round(stats.get("roi_x1_full", x1) - crop_x)))
                y1p = max(0, int(round(stats.get("roi_y1_full", y1) - crop_y)))
                mh, mw = mask.shape[:2]
                roi_overlay = overlay[y1p : y1p + mh, x1p : x1p + mw]
                if roi_overlay.shape[:2] == mask.shape[:2]:
                    tint = np.zeros_like(roi_overlay)
                    tint[:, :, 1] = 180
                    roi_overlay[mask > 0] = cv2.addWeighted(roi_overlay[mask > 0], 0.55, tint[mask > 0], 0.45, 0)
            cv2.putText(overlay, f"{det_id} n={len(points)} z={stats.get('z_near_mm', 0):.0f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            row = {
                "image": image_name,
                "det_id": det_id,
                "confidence": f"{float(det['confidence']):.6f}",
                "ok": int(ok),
                "point_count": len(points),
                "ply": str((ply_dir / ply_name).relative_to(PROJECT_ROOT)) if ok else "",
                "cx_mm": "" if centroid[0] is None else f"{centroid[0]:.3f}",
                "cy_mm": "" if centroid[1] is None else f"{centroid[1]:.3f}",
                "cz_mm": "" if centroid[2] is None else f"{centroid[2]:.3f}",
                "nx": "" if normal[0] is None else f"{normal[0]:.6f}",
                "ny": "" if normal[1] is None else f"{normal[1]:.6f}",
                "nz": "" if normal[2] is None else f"{normal[2]:.6f}",
                "z_near_mm": f"{stats.get('z_near_mm', 0):.3f}" if "z_near_mm" in stats else "",
                "z_far_mm": f"{stats.get('z_far_mm', 0):.3f}" if "z_far_mm" in stats else "",
                "component_count": stats.get("component_count", ""),
                "selected_component_area": stats.get("selected_component_area", ""),
            }
            rows.append(row)
            image_objects.append({**row, "centroid_mm": centroid, "normal": normal, "eigenvalues": eigenvalues, "det_xyxy": det["xyxy"]})
        json_records.append({"image": image_name, "objects": image_objects})
        if save_vis:
            cv2.imwrite(str(vis_dir / image_name), overlay)

    csv_path = out_dir / "object_clouds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "det_id", "confidence", "ok", "point_count", "ply", "cx_mm", "cy_mm", "cz_mm", "nx", "ny", "nz", "z_near_mm", "z_far_mm", "component_count", "selected_component_area"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = out_dir / "object_clouds.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in json_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok_count = sum(int(r["ok"]) for r in rows)
    print(f"{split}: detections={len(rows)} clouds_ok={ok_count} csv={csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract object-level RGB-D point clouds from detector boxes.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--detection-name", default="detections_pose")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--depth-window-mm", type=float, default=80.0)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(split, args.detection_name, args.class_name, args.depth_window_mm, args.min_points, save_vis=not args.no_vis)


if __name__ == "__main__":
    main()
