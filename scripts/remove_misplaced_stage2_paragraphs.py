from pathlib import Path

from docx import Document


DOC_PATH = Path.home() / "Desktop" / "论文.docx"


def remove_paragraph(paragraph) -> None:
    p = paragraph._element
    parent = p.getparent()
    if parent is not None:
        parent.remove(p)


def main() -> None:
    doc = Document(str(DOC_PATH))
    markers = [
        "二阶段遮挡恢复结果如下：",
        "训练集：总目标 479 个",
        "验证集：总目标 70 个",
        "测试集：总目标 57 个",
        "表格",
    ]
    for paragraph in list(doc.paragraphs):
        text = paragraph.text.strip()
        if any(text.startswith(marker) for marker in markers):
            # Keep the formal 4.12 paragraph, which starts without the colon-style list.
            if "验证集总目标 70 个" in text and "测试集总目标 57 个" in text:
                continue
            remove_paragraph(paragraph)
    doc.save(str(DOC_PATH))
    print(DOC_PATH)


if __name__ == "__main__":
    main()
