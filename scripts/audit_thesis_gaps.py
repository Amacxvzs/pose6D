from __future__ import annotations

from pathlib import Path

from docx import Document


DOC_PATH = Path.home() / "Desktop" / "论文.docx"
OUT_PATH = Path(r"D:\1.sjcl\pose6d\outputs\thesis_gap_report.md")


def main() -> None:
    doc = Document(str(DOC_PATH))
    lines: list[str] = [
        "# 论文待补实验与占位内容检查",
        "",
        "## 已经补入的真实实验内容",
        "",
        "- 3.5 基于 RGB-D 点云的二阶段遮挡恢复策略",
        "- 4.12 当前系统实现与阶段性实验结果",
        "- 图 4 二阶段 RGB-D 堆叠零件位姿估计流程",
        "- 图 5 二阶段遮挡恢复前后有效位姿输出统计",
        "- 图 6 测试集不同遮挡难度下的处理结果",
        "- 测试集二阶段最终可用位姿 53/57，可用率 93.0%，RMSE 均值 6.30 mm，IoU 均值 0.737",
        "",
        "## 仍需补充或核实的占位内容",
        "",
    ]
    placeholder_hits = []
    for idx, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if not text:
            continue
        if "【图" in text or text == "表格" or "注：使用" in text:
            placeholder_hits.append((idx, paragraph.style.name, text))
    if placeholder_hits:
        for idx, style, text in placeholder_hits:
            lines.append(f"- 段落 {idx}（{style}）：{text}")
    else:
        lines.append("- 未发现明显占位图或“表格”占位。")

    lines += [
        "",
        "## 建议优先完成的实验",
        "",
        "1. 检测模型正式训练与复现实验：固定 YOLOv9t 配置，保存训练日志、PR 曲线、混淆矩阵和 best.pt。",
        "2. 位姿评价指标完善：对当前盘类轴对称零件，优先统计中心误差、法向角误差、重投影 IoU、ICP RMSE 和可用率，不把 yaw 作为主指标。",
        "3. 二阶段遮挡恢复对比实验：比较无二阶段、只质量筛选、前景 mask 后二阶段恢复三种策略。",
        "4. 遮挡难度分组实验：继续使用无遮挡/轻微重叠/堆叠遮挡三类，补充更多测试图片以增强统计稳定性。",
        "5. 失败案例复查：对 skip_or_recapture 中的 4 个测试样本逐一截图并说明失败原因。",
        "6. 如果后续要保留原稿中的 CBAM/OAM、EPnP 角点、跨类别、跨数据集等章节，需要补真实实验或删改为展望/设计方案。",
        "",
        "## 当前建议",
        "",
        "短期论文主线建议改为：YOLOv9t 检测 + RGB-D 点云初始位姿 + CAD-ICP + 前景优先二阶段遮挡恢复。这样与已经完成的代码和实测数据一致，风险最低。",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
