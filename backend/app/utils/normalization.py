"""
实体名称归一化（Entity Normalization）。

背景问题（真实观察到的案例）：LLM 抽取三元组时，同一个实体可能被写成不同的形式，
比如 "120k~200k token" 和 "120k-200k token"——语义上是同一个东西，
但因为字符串不完全相等，MERGE 时会被 Neo4j 当成两个不同的节点，
导致图谱里出现看起来相似却没有连在一起的"孤立节点"，检索时也会漏掉一半的关联关系。

这一层做的是"轻量级"归一化，不是完整的实体消歧算法（真正严谨的实体消歧
通常需要向量相似度聚类或专门的实体链接模型），而是先解决几类最常见、
成本最低、收益最高的问题：
1. 首尾空格、内部连续空格
2. 各种"连接符"不统一（~、-、–、—、to 等都表示"范围"，统一成 "-"）
3. 全角/半角字符不统一
4. 可扩展的同义词词表（比如 "token" / "Token" 这种大小写不影响语义的场景）

这一层是可以持续迭代的地方——词表可以根据实际抽取效果不断补充。
"""
import re
import unicodedata

# 常见的"范围连接符"变体，统一替换成半角短横线 "-"
_RANGE_DASH_PATTERN = re.compile(r"[~～\-–—]|\bto\b")

# 同义词词表：键是"归一化后统一使用的形式"，值是会被替换成键的各种变体列表。
# 这是一个示例起点，实际使用中可以根据具体领域文档不断补充。
_SYNONYM_MAP: dict[str, list[str]] = {
    "token": ["tokens", "Token", "Tokens"],
}
_VARIANT_TO_CANONICAL = {
    variant: canonical
    for canonical, variants in _SYNONYM_MAP.items()
    for variant in variants
}


def normalize_entity_name(raw_name: str) -> str:
    """
    把一个实体名称归一化成"规范形式"，用于写入图谱前的最后一步清洗。

    注意：这个函数只做保守的、确定性的清洗，不改变实体的语义、
    不做模糊匹配或聚类（那是更复杂的实体消歧问题，属于已知的后续优化方向）。
    """
    if not raw_name:
        return raw_name

    name = raw_name.strip()

    # 全角字符转半角（比如全角括号、全角空格），中文文档里很常见
    name = unicodedata.normalize("NFKC", name)

    # 折叠连续空格为单个空格
    name = re.sub(r"\s+", " ", name)

    # 统一范围连接符：120k~200k / 120k-200k / 120k to 200k -> 统一成 120k-200k
    # 只在"数字...连接符...数字"这种明显是范围表达的场景做替换，避免误伤正常词汇里的连字符
    def _unify_range(match: re.Match) -> str:
        return "-"

    name = re.sub(
        r"(?<=[\dA-Za-z])\s*(?:[~～\-–—]|\bto\b)\s*(?=[\dA-Za-z])",
        "-",
        name,
    )

    # 应用同义词词表
    if name in _VARIANT_TO_CANONICAL:
        name = _VARIANT_TO_CANONICAL[name]

    return name.strip()
