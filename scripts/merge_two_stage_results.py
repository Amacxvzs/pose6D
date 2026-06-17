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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict) -> tuple[str, str]:
    return row["image"], str(row["det_id"])


def to_float(row: dict, field: str, default: float = 0.0) -> float:
    try:
        value = row.get(field, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {key(row): row for row in rows}


def merge_split(split: str) -> dict:
    main_poses = by_key(read_csv(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv"))
    stage2_poses = by_key(read_csv(PROJECT_ROOT / "outputs" / "icp_poses_stage2" / split / "icp_poses.csv"))
    stage1_plan = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "stage1_front_or_clear.csv")
    stage2_plan = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "stage2_occluded_candidates.csv")
    skip_plan = read_csv(PROJECT_ROOT / "outputs" / "occlusion_plan" / split / "skip_or_recapture.csv")

    rows: list[dict] = []
    for plan in stage1_plan:
        pose = main_poses.get(key(plan))
        if not pose:
            continue
        rows.append({**pose, "two_stage_source": "stage1_front_or_clear", "two_stage_valid": pose.get("valid", "0")})
    for plan in stage2_plan:
        pose = stage2_poses.get(key(plan))
        if not pose:
            continue
        rows.append({**pose, "two_stage_source": "stage2_after_front_mask", "two_stage_valid": pose.get("valid", "0")})

    out_dir = PROJECT_ROOT / "outputs" / "two_stage_poses" / split
    fieldnames = list(read_csv(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv")[0].keys()) + [
        "two_stage_source",
        "two_stage_valid",
    ]
    write_csv(out_dir / "two_stage_poses.csv", rows, fieldnames)

    valid_rows = [r for r in rows if str(r.get("two_stage_valid", "0")) == "1"]
    rmse = [to_float(r, "rmse_mm") for r in valid_rows if to_float(r, "rmse_mm") > 0]
    iou = [to_float(r, "final_iou") for r in valid_rows]
    center = [to_float(r, "final_center_error_px") for r in valid_rows]
    stage2_valid = sum(1 for r in rows if r["two_stage_source"] == "stage2_after_front_mask" and str(r.get("two_stage_valid", "0")) == "1")
    return {
        "split": split,
        "total_original": len(main_poses),
        "stage1_count": len(stage1_plan),
        "stage2_candidate_count": len(stage2_plan),
        "stage2_recovered": stage2_valid,
        "skip_or_recapture": len(skip_plan),
        "final_usable": len(valid_rows),
        "final_usable_rate": len(valid_rows) / len(main_poses) if main_poses else 0.0,
        "rmse_mean_mm": float(np.mean(rmse)) if rmse else 0.0,
        "rmse_median_mm": float(np.median(rmse)) if rmse else 0.0,
        "final_iou_mean": float(np.mean(iou)) if iou else 0.0,
        "center_error_mean_px": float(np.mean(center)) if center else 0.0,
    }


def fmt_pct(v: float) -> str:
    return f"{v * 100.0:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stage1 and stage2 pose results into final two-stage outputs.")
    parser.add_argument("--output", default="outputs/two_stage_summary.md")
    args = parser.parse_args()

    summary = [merge_split(split) for split in ("train", "val", "test")]
    out_path = PROJECT_ROOT / args.output
    csv_path = out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, summary, list(summary[0].keys()))

    lines = [
        "# Two-Stage Pose Summary",
        "",
        "| split | total | stage1 | stage2 candidates | stage2 recovered | skip/recapture | final usable | usable rate | RMSE mean (mm) | final IoU mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['split']} | {r['total_original']} | {r['stage1_count']} | {r['stage2_candidate_count']} | "
            f"{r['stage2_recovered']} | {r['skip_or_recapture']} | {r['final_usable']} | "
            f"{fmt_pct(r['final_usable_rate'])} | {r['rmse_mean_mm']:.2f} | {r['final_iou_mean']:.3f} |"
        )
    lines += [
        "",
        "Notes:",
        "- Stage1 uses clear or mild-overlap targets from the original RGB-D pose output.",
        "- Stage2 re-extracts rear-object point clouds after masking front blockers, then reruns initial pose and ICP.",
        "- Skip/recapture samples remain excluded from final usable statistics.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    print(csv_path)


if __name__ == "__main__":
    main()
