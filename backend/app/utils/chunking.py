"""
文本切块：把长文本切成适合 embedding 的小块。

策略说明（路线图里提到的坑）：
- 不能简单按固定字符数硬切，容易把一句话切断，损失语义完整性
- 这里采用"按句子分割 + 累积到接近 chunk_size 就切一刀"的策略
- overlap 让相邻块有一部分重叠内容，避免关键信息刚好卡在切割点丢失上下文
"""
import re


def split_into_sentences(text: str) -> list[str]:
    """按中文/英文标点做简单分句。"""
    # 在句号、问号、感叹号、分号后面切开，同时保留标点
    sentences = re.split(r"(?<=[。！？；\.\!\?])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    把文本切成多个 chunk。

    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的目标字符数
        chunk_overlap: 相邻 chunk 之间重叠的字符数

    Returns:
        切好的文本块列表
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        # 如果加上这句话会超过目标长度，且当前 chunk 已经有内容，就先切一刀
        if current_chunk and len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk)
            # 用 overlap 长度的尾部内容作为下一个 chunk 的开头，保持上下文连贯
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text + sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
