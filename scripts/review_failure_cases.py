from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(r"D:\1.sjcl\pose6d")
OUT_DIR = PROJECT_ROOT / "outputs" / "failure_review"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_detections(split: str = "test") -> dict[tuple[str, str], dict]:
    path = PROJECT_ROOT / "outputs" / "detections_pose" / split / "detections.jsonl"
    out: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for det in rec.get("detections", []):
                out[(rec["image"], str(det["det_id"]))] = {
                    "image": rec["image"],
                    "det_id": str(det["det_id"]),
                    "xyxy": [float(v) for v in det["xyxy"]],
                    "confidence": float(det["confidence"]),
                }
    return out


def key(row: dict) -> tuple[str, str]:
    return row["image"], str(row["det_id"])


def to_float(row: dict, field: str, default: float = 0.0) -> float:
    try:
        value = row.get(field, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def infer_reason(row: dict) -> str:
    if row.get("occlusion_level") == "occluded":
        return "前景遮挡且重投影/质量分数不足"
    if to_float(row, "final_iou") < 0.45:
        return "重投影 IoU 偏低，模型轮廓与检测框不一致"
    if to_float(row, "quality_score") < 50:
        return "质量分数低，点云或ICP结果不稳定"
    return "需人工复查"


def draw_case(image: np.ndarray, det: dict, row: dict) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
    label = f"det{det['det_id']} score={row['quality_score']} IoU={float(row['final_iou']):.2f}"
    cv2.putText(canvas, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if row.get("blocked_by_det_ids"):
        for bid in row["blocked_by_det_ids"].split(";"):
            bid = bid.strip()
            if not bid:
                continue
            # Caller overlays blocker only if present in detection dict.
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skip_rows = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / "test" / "skip_or_recapture.csv")
    quality_rows = {key(r): r for r in read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / "test" / "pose_quality.csv")}
    detections = load_detections("test")
    rgb_dir = PROJECT_ROOT / "data" / "raw" / "rgb" / "test" / "plate"

    review_rows: list[dict] = []
    panels: list[np.ndarray] = []
    for row in skip_rows:
        q = quality_rows.get(key(row), {})
        det = detections.get(key(row))
        image = cv2.imread(str(rgb_dir / row["image"]))
        if image is None or det is None:
            continue
        panel = image.copy()
        # Draw all detections lightly for context.
        for (img_name, det_id), other in detections.items():
            if img_name != row["image"]:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in other["xyxy"]]
            color = (0, 0, 255) if det_id == str(row["det_id"]) else (0, 190, 255)
            thickness = 3 if det_id == str(row["det_id"]) else 2
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(panel, f"id{det_id}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        reason = infer_reason(q)
        title = f"{Path(row['image']).stem[-14:]} det{row['det_id']}"
        panel = cv2.resize(panel, (360, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (360, 58), (255, 255, 255), -1)
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
        cv2.putText(panel, f"score={q.get('quality_score','')} IoU={q.get('final_iou','')[:5]}", (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
        panels.append(panel)

        review_rows.append(
            {
                "image": row["image"],
                "det_id": row["det_id"],
                "quality_score": q.get("quality_score", ""),
                "quality_level": q.get("quality_level", ""),
                "occlusion_level": q.get("occlusion_level", ""),
                "final_iou": q.get("final_iou", ""),
                "final_center_error_px": q.get("final_center_error_px", ""),
                "point_count": q.get("point_count", ""),
                "blocked_by_det_ids": row.get("blocked_by_det_ids", ""),
                "failure_reason": reason,
            }
        )

    if panels:
        cols = 2
        rows_img = []
        blank = np.full_like(panels[0], 255)
        for i in range(0, len(panels), cols):
            row_panels = panels[i : i + cols]
            if len(row_panels) < cols:
                row_panels += [blank] * (cols - len(row_panels))
            rows_img.append(np.hstack(row_panels))
        sheet = np.vstack(rows_img)
        cv2.imwrite(str(OUT_DIR / "fig_failure_cases.png"), sheet)

    with (OUT_DIR / "failure_cases.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "det_id",
            "quality_score",
            "quality_level",
            "occlusion_level",
            "final_iou",
            "final_center_error_px",
            "point_count",
            "blocked_by_det_ids",
            "failure_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    print(OUT_DIR / "failure_cases.csv")
    print(OUT_DIR / "fig_failure_cases.png")


if __name__ == "__main__":
    main()
