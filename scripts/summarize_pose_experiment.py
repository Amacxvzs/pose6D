from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def summarize_split(split: str) -> dict:
    icp = read_csv(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv")
    quality = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "pose_quality.csv")
    accepted = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "accepted_poses.csv")
    review = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "review_poses.csv")
    stage1 = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "stage1_front_or_clear.csv")
    stage2 = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "stage2_occluded_candidates.csv")
    skip = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "skip_or_recapture.csv")

    rmse = np.asarray([to_float(r, "rmse_mm") for r in icp if to_float(r, "rmse_mm") > 0], dtype=float)
    iou = np.asarray([to_float(r, "final_iou") for r in icp], dtype=float)
    center = np.asarray([to_float(r, "final_center_error_px") for r in icp], dtype=float)
    scores = np.asarray([to_float(r, "quality_score") for r in quality], dtype=float)

    def mean_or_zero(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else 0.0

    reliable = sum(1 for r in quality if r.get("quality_level") == "reliable")
    usable = sum(1 for r in quality if r.get("quality_level") == "usable")
    low = sum(1 for r in quality if r.get("quality_level") == "low_confidence")

    return {
        "split": split,
        "total_poses": len(icp),
        "accepted": len(accepted),
        "review": len(review),
        "reliable": reliable,
        "usable": usable,
        "low_confidence": low,
        "stage1_front_or_clear": len(stage1),
        "stage2_occluded_candidates": len(stage2),
        "skip_or_recapture": len(skip),
        "accept_rate": len(accepted) / len(icp) if icp else 0.0,
        "rmse_mean_mm": mean_or_zero(rmse),
        "rmse_median_mm": float(np.median(rmse)) if rmse.size else 0.0,
        "final_iou_mean": mean_or_zero(iou),
        "center_error_mean_px": mean_or_zero(center),
        "quality_score_mean": mean_or_zero(scores),
    }


def fmt_pct(v: float) -> str:
    return f"{v * 100.0:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pose estimation and occlusion-plan results.")
    parser.add_argument("--output", default="outputs/pose_experiment_summary.md")
    args = parser.parse_args()

    rows = [summarize_split(split) for split in ("train", "val", "test")]
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Pose Experiment Summary",
        "",
        "## Overall",
        "",
        "| split | total | accepted | review | accept rate | reliable | usable | low confidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['total_poses']} | {r['accepted']} | {r['review']} | "
            f"{fmt_pct(r['accept_rate'])} | {r['reliable']} | {r['usable']} | {r['low_confidence']} |"
        )

    lines += [
        "",
        "## Occlusion Plan",
        "",
        "| split | stage1 front/clear | stage2 occluded candidates | skip/recapture |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['stage1_front_or_clear']} | "
            f"{r['stage2_occluded_candidates']} | {r['skip_or_recapture']} |"
        )

    lines += [
        "",
        "## Pose Metrics",
        "",
        "| split | rmse mean (mm) | rmse median (mm) | final IoU mean | center error mean (px) | quality score mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['rmse_mean_mm']:.2f} | {r['rmse_median_mm']:.2f} | "
            f"{r['final_iou_mean']:.3f} | {r['center_error_mean_px']:.2f} | {r['quality_score_mean']:.1f} |"
        )
    lines += [
        "",
        "Notes:",
        "- Accepted poses contain reliable poses plus usable poses without front-object blocking.",
        "- Stage2 candidates are rear or overlapped objects that should be processed after masking/accepting front objects.",
        "- Low-confidence cases should be excluded from the main quantitative table and discussed as difficult/failure samples.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    print(csv_path)


if __name__ == "__main__":
    main()
