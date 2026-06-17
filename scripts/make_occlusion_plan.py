from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def load_detections(split: str, detection_name: str) -> dict[tuple[str, str], dict]:
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
                    "xyxy": [float(v) for v in det["xyxy"]],
                }
    return out


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def group_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["image"], []).append(row)
    return out


def run(split: str, detection_name: str, z_margin_mm: float, min_overlap: float) -> None:
    quality_rows = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "pose_quality.csv")
    detections = load_detections(split, detection_name)
    by_image = group_by_image(quality_rows)

    order_rows: list[dict] = []
    stage1_rows: list[dict] = []
    rear_rows: list[dict] = []
    skip_rows: list[dict] = []

    for image, rows in by_image.items():
        rows = sorted(rows, key=lambda r: (to_float(r, "z_near_mm", 999999.0), int(r["det_id"])))
        for rank, row in enumerate(rows, start=1):
            key = (row["image"], row["det_id"])
            det = detections.get(key)
            z = to_float(row, "z_near_mm", 0.0)
            blockers: list[str] = []
            blocker_ious: list[str] = []
            if det:
                for other in rows:
                    if other["det_id"] == row["det_id"]:
                        continue
                    other_det = detections.get((other["image"], other["det_id"]))
                    if not other_det:
                        continue
                    other_z = to_float(other, "z_near_mm", 0.0)
                    iou = bbox_iou(det["xyxy"], other_det["xyxy"])
                    if iou >= min_overlap and z > other_z + z_margin_mm:
                        blockers.append(other["det_id"])
                        blocker_ious.append(f"{other['det_id']}:{iou:.4f}")

            if row["quality_level"] == "low_confidence":
                stage = "skip_or_recapture"
                skip_rows.append(row)
            elif blockers:
                stage = "stage2_after_front_mask"
                rear_rows.append(row)
            else:
                stage = "stage1_front_or_clear"
                stage1_rows.append(row)

            order_rows.append(
                {
                    "image": image,
                    "det_id": row["det_id"],
                    "front_to_back_rank": rank,
                    "stage": stage,
                    "blocked_by_det_ids": ";".join(blockers),
                    "blocked_by_iou": ";".join(blocker_ious),
                    "z_near_mm": row["z_near_mm"],
                    "quality_score": row["quality_score"],
                    "quality_level": row["quality_level"],
                    "occlusion_level": row["occlusion_level"],
                    "recommended_action": row["recommended_action"],
                    "final_iou": row["final_iou"],
                    "final_center_error_px": row["final_center_error_px"],
                    "point_count": row["point_count"],
                }
            )

    out_dir = PROJECT_ROOT / "outputs" / "occlusion_plan" / split
    fields = [
        "image",
        "det_id",
        "front_to_back_rank",
        "stage",
        "blocked_by_det_ids",
        "blocked_by_iou",
        "z_near_mm",
        "quality_score",
        "quality_level",
        "occlusion_level",
        "recommended_action",
        "final_iou",
        "final_center_error_px",
        "point_count",
    ]
    write_csv(out_dir / "stack_order.csv", order_rows, fields)
    write_csv(out_dir / "stage1_front_or_clear.csv", [r for r in order_rows if r["stage"] == "stage1_front_or_clear"], fields)
    write_csv(out_dir / "stage2_occluded_candidates.csv", [r for r in order_rows if r["stage"] == "stage2_after_front_mask"], fields)
    write_csv(out_dir / "skip_or_recapture.csv", [r for r in order_rows if r["stage"] == "skip_or_recapture"], fields)
    print(
        f"{split}: total={len(order_rows)} "
        f"stage1={len(stage1_rows)} stage2={len(rear_rows)} skip={len(skip_rows)} "
        f"-> {out_dir / 'stack_order.csv'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a front-to-back occlusion handling plan from pose quality outputs.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--detection-name", default="detections_pose")
    parser.add_argument("--z-margin-mm", type=float, default=8.0)
    parser.add_argument("--min-overlap", type=float, default=0.03)
    args = parser.parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        run(split, args.detection_name, args.z_margin_mm, args.min_overlap)


if __name__ == "__main__":
    main()
