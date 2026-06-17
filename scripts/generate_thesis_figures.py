from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


PROJECT_ROOT = Path(r"D:\1.sjcl\pose6d")
OUT_DIR = PROJECT_ROOT / "outputs" / "thesis_figures"


def pick_font() -> str | None:
    keywords = ("msyh", "simhei", "simsun", "noto")
    for font_path in font_manager.findSystemFonts():
        lower = font_path.lower()
        if any(k in lower for k in keywords):
            return font_path
    return None


def setup_font() -> None:
    font_path = pick_font()
    if font_path:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_axis_off()
    steps = [
        ("RGB-D采集", "RGB全图\n16位深度"),
        ("YOLOv9t检测", "盘类零件\n2D检测框"),
        ("坐标还原", "裁剪坐标\n映射全图"),
        ("点云提取", "深度窗口\n连通域筛选"),
        ("初始位姿", "点云质心\nPCA法向"),
        ("CAD-ICP", "CAD采样\n点云精修"),
        ("质量评估", "IoU/RMSE\n遮挡关系"),
        ("二阶段恢复", "前景mask\n后景重估"),
    ]
    coords = [
        (0.14, 0.68),
        (0.38, 0.68),
        (0.62, 0.68),
        (0.86, 0.68),
        (0.86, 0.34),
        (0.62, 0.34),
        (0.38, 0.34),
        (0.14, 0.34),
    ]
    box_w, box_h = 0.17, 0.20
    for i, (title, subtitle) in enumerate(steps):
        x, y = coords[i]
        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#245b7a",
            facecolor="#eaf4f8" if i < 6 else "#fff3d6",
        )
        ax.add_patch(patch)
        ax.text(x, y + 0.045, title, ha="center", va="center", fontsize=10.5, weight="bold", color="#15384f")
        ax.text(x, y - 0.052, subtitle, ha="center", va="center", fontsize=8.8, color="#3a4a56", linespacing=1.35)
        if i < len(steps) - 1:
            x2, y2 = coords[i + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (x + (box_w / 2 + 0.01) * np.sign(x2 - x), y if y == y2 else y - box_h / 2 - 0.01),
                    (x2 - (box_w / 2 + 0.01) * np.sign(x2 - x), y2 if y == y2 else y2 + box_h / 2 + 0.01),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.1,
                    color="#4c6f80",
                )
            )
    ax.text(0.5, 0.94, "二阶段RGB-D堆叠零件位姿估计流程", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.11, "核心思想：先估计前景可见件，再剔除前景深度区域并恢复后景遮挡件", ha="center", fontsize=10.5, color="#555555")
    return save(fig, "fig_rgbd_two_stage_pipeline.png")


def draw_two_stage_summary() -> Path:
    rows = read_csv(PROJECT_ROOT / "outputs" / "two_stage_summary.csv")
    labels = ["训练集", "验证集", "测试集"]
    total = np.array([float(r["total_original"]) for r in rows])
    stage1 = np.array([float(r["stage1_count"]) for r in rows])
    stage2 = np.array([float(r["stage2_recovered"]) for r in rows])
    skip = np.array([float(r["skip_or_recapture"]) for r in rows])
    usable_rate = np.array([float(r["final_usable_rate"]) * 100 for r in rows])

    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, stage1, label="第一阶段可用", color="#4c9f70")
    ax.bar(x, stage2, bottom=stage1, label="二阶段恢复", color="#f2b84b")
    ax.bar(x, skip, bottom=stage1 + stage2, label="跳过/重拍", color="#d75a4a")
    for i, rate in enumerate(usable_rate):
        ax.text(i, total[i] + max(total) * 0.025, f"可用率 {rate:.1f}%", ha="center", fontsize=9.5, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("目标数量")
    ax.set_title("二阶段遮挡恢复前后有效位姿输出统计", fontsize=13, weight="bold")
    ax.set_ylim(0, max(total) * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    return save(fig, "fig_two_stage_summary.png")


def draw_difficulty_summary() -> Path:
    rows = [r for r in read_csv(PROJECT_ROOT / "outputs" / "pose_difficulty_summary.csv") if r["split"] == "test"]
    labels = ["无遮挡/分离", "轻微重叠", "堆叠遮挡"]
    totals = np.array([float(r["total"]) for r in rows])
    accepted = np.array([float(r["accepted"]) for r in rows])
    stage2 = np.array([float(r["stage2_candidates"]) for r in rows])
    low = np.array([float(r["low_confidence"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, accepted, label="直接可用", color="#4c9f70")
    ax.bar(x, stage2, bottom=accepted, label="二阶段候选", color="#f2b84b")
    ax.bar(x, low, bottom=accepted + stage2, label="低置信", color="#d75a4a")
    for i, total in enumerate(totals):
        ax.text(i, total + 0.6, f"n={int(total)}", ha="center", fontsize=9.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("测试集目标数量")
    ax.set_title("测试集不同遮挡难度下的处理结果", fontsize=13, weight="bold")
    ax.set_ylim(0, max(totals) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    return save(fig, "fig_difficulty_summary.png")


def main() -> None:
    setup_font()
    paths = [draw_pipeline(), draw_two_stage_summary(), draw_difficulty_summary()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
