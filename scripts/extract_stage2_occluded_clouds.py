from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from extract_rgbd_object_clouds import extract_one, load_intrinsic, load_meta, pca_normal, write_ply


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_detections(split: str, detection_name: str) -> dict[tuple[str, str], dict]:
    path = PROJECT_ROOT / "outputs" / detection_name / split / "detections.jsonl"
    out: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for det in record.get("detections", []):
                det_id = str(det["det_id"])
                out[(record["image"], det_id)] = {
                    "image": record["image"],
                    "det_id": det_id,
                    "confidence": float(det["confidence"]),
                    "xyxy": [float(v) for v in det["xyxy"]],
                }
    return out


def parse_blockers(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def mask_blockers(
    depth_full: np.ndarray,
    image_name: str,
    blocker_ids: list[str],
    detections: dict[tuple[str, str], dict],
    crop_x: float,
    crop_y: float,
    margin_px: int,
) -> int:
    h, w = depth_full.shape[:2]
    masked_pixels = 0
    for blocker_id in blocker_ids:
        det = detections.get((image_name, blocker_id))
        if not det:
            continue
        x1, y1, x2, y2 = det["xyxy"]
        x1 = max(0, int(round(x1 + crop_x)) - margin_px)
        y1 = max(0, int(round(y1 + crop_y)) - margin_px)
        x2 = min(w - 1, int(round(x2 + crop_x)) + margin_px)
        y2 = min(h - 1, int(round(y2 + crop_y)) + margin_px)
        if x2 < x1 or y2 < y1:
            continue
        masked_pixels += int(np.count_nonzero(depth_full[y1 : y2 + 1, x1 : x2 + 1]))
        depth_full[y1 : y2 + 1, x1 : x2 + 1] = 0
    return masked_pixels


def run(
    split: str,
    class_name: str,
    detection_name: str,
    depth_window_mm: float,
    min_points: int,
    blocker_margin_px: int,
) -> None:
    fx, fy, cx, cy = load_intrinsic(PROJECT_ROOT / "data" / "calibration" / "camera_intrinsic.json")
    candidates = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "stage2_occluded_candidates.csv")
    detections = read_detections(split, detection_name)

    rgb_full_dir = PROJECT_ROOT / "data" / "raw" / "rgb_full" / split / class_name
    depth_dir = PROJECT_ROOT / "data" / "raw" / "depth" / split / class_name
    meta_dir = PROJECT_ROOT / "data" / "raw" / "meta" / split / class_name
    out_dir = PROJECT_ROOT / "outputs" / "object_clouds_stage2" / split
    ply_dir = out_dir / "ply"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    records: dict[str, list[dict]] = {}
    for candidate in candidates:
        image_name = candidate["image"]
        det_id = str(candidate["det_id"])
        det = detections.get((image_name, det_id))
        if not det:
            continue
        stem = Path(image_name).stem
        rgb_full = cv2.imread(str(rgb_full_dir / image_name), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_dir / image_name), cv2.IMREAD_UNCHANGED)
        if rgb_full is None or depth is None:
            continue

        meta = load_meta(meta_dir / f"{stem}.json")
        crop_x = float(meta.get("rgb_crop_x", 0.0))
        crop_y = float(meta.get("rgb_crop_y", 0.0))
        depth_full = cv2.resize(depth, (rgb_full.shape[1], rgb_full.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth_full = depth_full.copy()
        blocker_ids = parse_blockers(candidate.get("blocked_by_det_ids", ""))
        masked_pixels = mask_blockers(depth_full, image_name, blocker_ids, detections, crop_x, crop_y, blocker_margin_px)

        points, colors, _mask, stats = extract_one(
            rgb_full,
            depth_full,
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
        ply_name = f"{stem}_det{int(det_id):02d}.ply"
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

        row = {
            "image": image_name,
            "det_id": det_id,
            "confidence": f"{det['confidence']:.6f}",
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
            "blocked_by_det_ids": ";".join(blocker_ids),
            "masked_depth_pixels": masked_pixels,
        }
        rows.append(row)
        records.setdefault(image_name, []).append(
            {**row, "centroid_mm": centroid, "normal": normal, "eigenvalues": eigenvalues, "det_xyxy": det["xyxy"]}
        )

    fieldnames = [
        "image",
        "det_id",
        "confidence",
        "ok",
        "point_count",
        "ply",
        "cx_mm",
        "cy_mm",
        "cz_mm",
        "nx",
        "ny",
        "nz",
        "z_near_mm",
        "z_far_mm",
        "component_count",
        "selected_component_area",
        "blocked_by_det_ids",
        "masked_depth_pixels",
    ]
    with (out_dir / "object_clouds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "object_clouds.jsonl").open("w", encoding="utf-8") as f:
        for image_name, objects in records.items():
            f.write(json.dumps({"image": image_name, "objects": objects}, ensure_ascii=False) + "\n")

    ok_count = sum(int(r["ok"]) for r in rows)
    print(f"{split}: stage2_candidates={len(candidates)} clouds_ok={ok_count} -> {out_dir / 'object_clouds.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-extract occluded rear-object clouds after masking front blockers.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--detection-name", default="detections_pose")
    parser.add_argument("--depth-window-mm", type=float, default=80.0)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--blocker-margin-px", type=int, default=6)
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(split, args.class_name, args.detection_name, args.depth_window_mm, args.min_points, args.blocker_margin_px)


if __name__ == "__main__":
    main()
