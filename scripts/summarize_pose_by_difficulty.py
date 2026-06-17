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


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def row_key(row: dict) -> tuple[str, str]:
    return row["image"], str(row["det_id"])


def summarize_group(split: str, name: str, rows: list[dict], pose_by_key: dict[tuple[str, str], dict]) -> dict:
    total = len(rows)
    accepted = [r for r in rows if r.get("recommended_action") in {"use_pose", "use_with_caution"}]
    stage2 = [r for r in rows if r.get("recommended_action") == "estimate_front_objects_first"]
    low = [r for r in rows if r.get("quality_level") == "low_confidence"]
    accepted_pose_rows = [pose_by_key[row_key(r)] for r in accepted if row_key(r) in pose_by_key]
    rmse = [to_float(r, "rmse_mm") for r in accepted_pose_rows if to_float(r, "rmse_mm") > 0]
    iou = [to_float(r, "final_iou") for r in accepted_pose_rows]
    center = [to_float(r, "final_center_error_px") for r in accepted_pose_rows]
    score = [to_float(r, "quality_score") for r in rows]
    return {
        "split": split,
        "difficulty": name,
        "total": total,
        "accepted": len(accepted),
        "stage2_candidates": len(stage2),
        "low_confidence": len(low),
        "accept_rate": len(accepted) / total if total else 0.0,
        "rmse_mean_mm": mean_or_zero(rmse),
        "rmse_median_mm": median_or_zero(rmse),
        "final_iou_mean": mean_or_zero(iou),
        "center_error_mean_px": mean_or_zero(center),
        "quality_score_mean": mean_or_zero(score),
    }


def summarize_split(split: str) -> list[dict]:
    quality = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "pose_quality.csv")
    pose_rows = read_csv(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv")
    pose_by_key = {row_key(r): r for r in pose_rows}
    groups = {
        "clear_single_or_separated": [r for r in quality if r.get("occlusion_level") == "clear"],
        "partial_overlap": [r for r in quality if r.get("occlusion_level") == "partial"],
        "occluded_stack": [r for r in quality if r.get("occlusion_level") == "occluded"],
    }
    return [summarize_group(split, name, rows, pose_by_key) for name, rows in groups.items()]


def fmt_pct(v: float) -> str:
    return f"{v * 100.0:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pose results by occlusion difficulty.")
    parser.add_argument("--output", default="outputs/pose_difficulty_summary.md")
    args = parser.parse_args()

    rows: list[dict] = []
    for split in ("train", "val", "test"):
        rows.extend(summarize_split(split))

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Pose Difficulty Summary",
        "",
        "| split | difficulty | total | accepted | stage2 | low confidence | accept rate | RMSE mean (mm) | final IoU mean | center error mean (px) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['difficulty']} | {r['total']} | {r['accepted']} | "
            f"{r['stage2_candidates']} | {r['low_confidence']} | {fmt_pct(r['accept_rate'])} | "
            f"{r['rmse_mean_mm']:.2f} | {r['final_iou_mean']:.3f} | {r['center_error_mean_px']:.2f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- clear_single_or_separated: no obvious box overlap or front-object blocking.",
        "- partial_overlap: neighboring boxes overlap or multiple depth components appear, but no front object clearly blocks the target.",
        "- occluded_stack: a nearer object overlaps the target; these samples should be handled with front-to-back processing or discussed as stacked occlusion cases.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    print(csv_path)


if __name__ == "__main__":
    main()
