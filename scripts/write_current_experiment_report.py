from __future__ import annotations

from pathlib import Path
import csv


PROJECT_ROOT = Path(r"D:\1.sjcl\pose6d")
OUT = PROJECT_ROOT / "outputs" / "current_experiment_report.md"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    two = {r["split"]: r for r in read_csv(PROJECT_ROOT / "outputs" / "two_stage_summary.csv")}
    abl = [r for r in read_csv(PROJECT_ROOT / "outputs" / "ablation_experiment" / "ablation_summary.csv") if r["split"] == "test"]
    fail = read_csv(PROJECT_ROOT / "outputs" / "failure_review" / "failure_cases.csv")
    test = two["test"]

    lines = [
        "# 当前实验进展报告",
        "",
        "## 已完成实验",
        "",
        "1. 数据采集与划分：完成 260 组盘类零件 RGB-D 样本采集，划分为训练集 200 组、验证集 30 组、测试集 30 组。",
        "2. 检测模型训练：完成 YOLOv9t 检测训练，测试集 mAP50=0.973，mAP50-95=0.868。",
        "3. RGB-D 点云位姿估计：完成检测框坐标还原、深度点云提取、PCA 初始位姿、CAD-ICP 精修。",
        "4. 质量评估：完成 reliable / usable / low_confidence 分类，输出 accepted_poses 与 review_poses。",
        "5. 二阶段遮挡恢复：完成前景 mask 后的后景点云重提取，遮挡候选重新估计位姿。",
        "6. 失败案例复查：完成 4 个低置信样本的图像与原因表整理。",
        "",
        "## 核心结果",
        "",
        f"- 测试集总目标数：{test['total_original']}",
        f"- 第一阶段可用目标：{test['stage1_count']}",
        f"- 二阶段遮挡候选：{test['stage2_candidate_count']}",
        f"- 二阶段恢复目标：{test['stage2_recovered']}",
        f"- 最终可用目标：{test['final_usable']}",
        f"- 最终可用率：{float(test['final_usable_rate']) * 100:.1f}%",
        f"- RMSE 均值：{float(test['rmse_mean_mm']):.2f} mm",
        f"- 最终 IoU 均值：{float(test['final_iou_mean']):.3f}",
        "",
        "## 测试集后处理策略消融",
        "",
        "| 策略 | 可用目标 | 可用率 | RMSE均值/mm | IoU均值 |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {
        "raw_icp_all_valid": "原始ICP全部输出",
        "quality_filter_only": "仅质量筛选",
        "quality_including_stage2_candidates": "质量筛选+遮挡候选",
        "two_stage_front_mask": "二阶段前景mask恢复",
    }
    for r in abl:
        lines.append(
            f"| {names.get(r['strategy'], r['strategy'])} | {r['usable_targets']} | "
            f"{float(r['usable_rate']) * 100:.1f}% | {float(r['rmse_mean_mm']):.2f} | {float(r['final_iou_mean']):.3f} |"
        )

    lines += [
        "",
        "## 失败案例",
        "",
        "| 图像 | det_id | 质量分数 | IoU | 原因 |",
        "|---|---:|---:|---:|---|",
    ]
    for r in fail:
        lines.append(
            f"| {r['image']} | {r['det_id']} | {r['quality_score']} | {float(r['final_iou']):.3f} | {r['failure_reason']} |"
        )

    lines += [
        "",
        "## 已生成论文图",
        "",
        "- 图 1：二阶段 RGB-D 堆叠零件位姿估计流程",
        "- 图 2：测试集不同遮挡难度下的处理结果",
        "- 图 3：低置信失败案例示例",
        "- 图 4：二阶段遮挡恢复前后有效位姿输出统计",
        "- 图 5：测试集不同后处理策略消融对比",
        "",
        "## 下一步建议",
        "",
        "1. 增加测试集规模，尤其是堆叠遮挡样本，提升 11 个遮挡目标统计的可信度。",
        "2. 为二阶段恢复补充法向角误差或人工可用性评价，避免只依赖 IoU 和 RMSE。",
        "3. 将论文主线收敛到 YOLOv9t + RGB-D 点云 + CAD-ICP + 二阶段遮挡恢复，删除或降级尚未实测的 OAM/四元数辅助损失等内容。",
    ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
