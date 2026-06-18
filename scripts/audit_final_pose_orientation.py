from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normal_tilt_deg(row: dict) -> float:
    rvec = np.asarray([float(row["rx"]), float(row["ry"]), float(row["rz"])], dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    return math.degrees(math.acos(np.clip(abs(float(normal[2])), 0.0, 1.0)))


def run(split: str, max_tilt_deg: float) -> dict:
    poses = read_csv(PROJECT_ROOT / "outputs" / "two_stage_poses" / split / "two_stage_poses.csv")
    skips = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "skip_or_recapture.csv")
    rows: list[dict] = []
    for pose in poses:
        tilt = normal_tilt_deg(pose)
        orientation_valid = tilt <= max_tilt_deg
        rows.append(
            {
                "image": pose["image"],
                "det_id": pose["det_id"],
                "source": pose["two_stage_source"],
                "final_iou": pose["final_iou"],
                "center_error_px": pose["final_center_error_px"],
                "rmse_mm": pose["rmse_mm"],
                "normal_tilt_deg": f"{tilt:.3f}",
                "position_valid": pose["two_stage_valid"],
                "orientation_valid": int(orientation_valid),
                "strict_pose_valid": int(str(pose["two_stage_valid"]) == "1" and orientation_valid),
            }
        )

    out_dir = PROJECT_ROOT / "outputs" / "final_pose_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{split}_pose_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(poses) + len(skips)
    position_valid = sum(int(r["position_valid"]) for r in rows)
    orientation_outliers = sum(not int(r["orientation_valid"]) for r in rows)
    strict_valid = sum(int(r["strict_pose_valid"]) for r in rows)
    return {
        "split": split,
        "total": total,
        "position_valid": position_valid,
        "position_valid_rate": position_valid / total if total else 0.0,
        "orientation_outliers": orientation_outliers,
        "strict_valid": strict_valid,
        "strict_valid_rate": strict_valid / total if total else 0.0,
        "skip": len(skips),
        "max_tilt_deg": max_tilt_deg,
        "csv": csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final pose orientation for an approximately horizontal plate.")
    parser.add_argument("--max-tilt-deg", type=float, default=35.0)
    args = parser.parse_args()

    summaries = [run(split, args.max_tilt_deg) for split in ("train", "val", "test")]
    output = PROJECT_ROOT / "outputs" / "final_pose_audit" / "summary.md"
    lines = [
        "# Final Pose Orientation Audit",
        "",
        f"Normal tilt threshold: {args.max_tilt_deg:.1f} deg",
        "",
        "| split | total | position valid | position rate | orientation outliers | strict valid | strict rate | skip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['total']} | {row['position_valid']} | "
            f"{row['position_valid_rate'] * 100:.1f}% | {row['orientation_outliers']} | "
            f"{row['strict_valid']} | {row['strict_valid_rate'] * 100:.1f}% | {row['skip']} |"
        )
    lines += [
        "",
        "Position valid means the projected circle/center and ICP checks passed.",
        "Strict valid additionally requires the plate normal to remain within the tilt threshold.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    for row in summaries:
        print(row["csv"])


if __name__ == "__main__":
    main()
