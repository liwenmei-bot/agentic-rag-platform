"""
文档解析：把不同格式的文件统一转成纯文本。
后续如果要支持更复杂的格式（扫描版 PDF、图片 OCR），可以在这里扩展，
接口签名（输入文件路径，输出字符串）保持不变即可。
"""
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def parse_document(file_path: str) -> str:
    """根据文件后缀选择对应的解析方式，返回纯文本内容。"""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_path)
    elif suffix == ".docx":
        return _parse_docx(file_path)
    elif suffix == ".txt":
        return _parse_txt(file_path)
    else:
        raise ValueError(f"暂不支持的文件类型: {suffix}，目前仅支持 pdf / docx / txt")


def _parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def _parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
