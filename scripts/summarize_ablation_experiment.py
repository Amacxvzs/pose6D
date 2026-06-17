from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(r"D:\1.sjcl\pose6d")
OUT_DIR = PROJECT_ROOT / "outputs" / "ablation_experiment"


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


def setup_font() -> None:
    for font_path in font_manager.findSystemFonts():
        lower = font_path.lower()
        if any(k in lower for k in ("msyh", "simhei", "simsun", "noto")):
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_split(split: str) -> list[dict]:
    icp = read_csv(PROJECT_ROOT / "outputs" / "icp_poses" / split / "icp_poses.csv")
    quality = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "pose_quality.csv")
    accepted = read_csv(PROJECT_ROOT / "outputs" / "pose_quality" / split / "accepted_poses.csv")
    two_stage = read_csv(PROJECT_ROOT / "outputs" / "two_stage_poses" / split / "two_stage_poses.csv")
    total = len(icp)

    def metrics(name: str, rows: list[dict], valid_field: str = "valid") -> dict:
        valid_rows = [r for r in rows if str(r.get(valid_field, "1")) == "1"]
        rmse = [to_float(r, "rmse_mm") for r in valid_rows if to_float(r, "rmse_mm") > 0]
        iou = [to_float(r, "final_iou") for r in valid_rows]
        center = [to_float(r, "final_center_error_px") for r in valid_rows]
        return {
            "split": split,
            "strategy": name,
            "total_targets": total,
            "usable_targets": len(valid_rows),
            "usable_rate": len(valid_rows) / total if total else 0.0,
            "rmse_mean_mm": mean(rmse),
            "final_iou_mean": mean(iou),
            "center_error_mean_px": mean(center),
        }

    reliable_or_usable = [r for r in quality if r.get("recommended_action") in {"use_pose", "use_with_caution", "estimate_front_objects_first"}]
    return [
        metrics("raw_icp_all_valid", icp),
        metrics("quality_filter_only", accepted),
        metrics("quality_including_stage2_candidates", reliable_or_usable),
        metrics("two_stage_front_mask", two_stage, valid_field="two_stage_valid"),
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_test_chart(rows: list[dict]) -> Path:
    setup_font()
    test_rows = [r for r in rows if r["split"] == "test"]
    labels = ["原始ICP", "质量筛选", "含遮挡候选", "二阶段恢复"]
    usable_rates = [float(r["usable_rate"]) * 100 for r in test_rows]
    rmse = [float(r["rmse_mean_mm"]) for r in test_rows]

    fig, ax1 = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(labels))
    bars = ax1.bar(x, usable_rates, color=["#8aa6c1", "#4c9f70", "#f2b84b", "#2f7f9f"], label="可用率")
    ax1.set_ylabel("可用率 / %")
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, usable_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9.5)

    ax2 = ax1.twinx()
    ax2.plot(x, rmse, color="#d75a4a", marker="o", linewidth=2, label="RMSE均值")
    ax2.set_ylabel("RMSE均值 / mm")
    ax2.set_ylim(0, max(rmse) * 1.5 if rmse else 10)
    for xi, value in zip(x, rmse):
        ax2.text(xi, value + 0.25, f"{value:.2f}", color="#a33b32", ha="center", fontsize=9)

    ax1.set_title("测试集不同后处理策略消融对比", fontsize=13, weight="bold")
    lines, names = ax1.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, names + names2, loc="lower right", frameon=False)

    path = OUT_DIR / "fig_ablation_test_strategies.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# 二阶段遮挡恢复消融实验",
        "",
        "| split | strategy | total | usable | usable rate | RMSE mean (mm) | final IoU mean | center error mean (px) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['strategy']} | {r['total_targets']} | {r['usable_targets']} | "
            f"{float(r['usable_rate']) * 100:.1f}% | {float(r['rmse_mean_mm']):.2f} | "
            f"{float(r['final_iou_mean']):.3f} | {float(r['center_error_mean_px']):.2f} |"
        )
    lines += [
        "",
        "说明：quality_filter_only 仅保留可靠和谨慎可用样本；two_stage_front_mask 在遮挡候选中进一步执行前景深度区域剔除和后景点云重估。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        rows.extend(summarize_split(split))
    write_csv(OUT_DIR / "ablation_summary.csv", rows)
    write_markdown(OUT_DIR / "ablation_summary.md", rows)
    fig = draw_test_chart(rows)
    print(OUT_DIR / "ablation_summary.csv")
    print(OUT_DIR / "ablation_summary.md")
    print(fig)


if __name__ == "__main__":
    main()
