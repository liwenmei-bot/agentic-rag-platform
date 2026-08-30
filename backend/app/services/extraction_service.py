"""
三元组抽取服务。

目标：
1. 只抽取原文明确支持的事实，不补写、不推测。
2. 实体尽量短、稳定、可复用，避免整句被当成实体。
3. 同一 chunk 内对实体命名做保守归一化，减少 Agent/agent 等重复节点。
4. 对 LLM 输出做代码侧校验、清洗和去重，避免脏三元组直接进入 Neo4j。
"""

import json
import re

from openai import OpenAI

from app.core.config import settings

_llm_client = OpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)


EXTRACTION_PROMPT = """你是一个严格的知识图谱信息抽取器。
请从【文本内容】中抽取少量、高质量、可复用的实体关系三元组。

必须遵守：
1. 只抽取文本中明确表达的事实关系。禁止推测、补全、常识扩写或为了建图而创造关系。
2. 每个三元组必须包含 subject、relation、object 三个字段。
3. subject 和 object 必须尽量是“短而稳定的实体/概念名称”，不要直接复制整句话。
4. relation 必须是简短、明确的关系名称，优先 2~6 个字，例如：定义、包含、属于、依赖、导致、限制、执行方式、衡量单位。
5. 同一个概念有不同大小写时统一名称。例如 Agent/agent 统一成 Agent；Neo4j/neo4j 统一成 Neo4j；RAG/rag 统一成 RAG。
6. 不要擅自把不同概念合并。例如“Agent”“Agent 工程”“智能体”只有在原文明确表示它们是同一概念时才能统一。
7. 如果文本明确表达“X 通过多轮执行……”，可以抽取 X -[执行方式]-> 多轮执行；如果原文没有表达，就不能生成这条关系。
8. 如果一个定义性客体较长，应压缩成不改变原意的短概念短语；不要加入原文没有的术语。
9. 最多抽取 10 条最重要的三元组。宁可少抽，也不要抽无意义或低置信度关系。
10. 不要输出自环关系（subject 与 object 相同）。
11. 只输出 JSON 数组，不要输出 Markdown、解释、前后缀或其他文字。

输出格式示例：
[{"subject":"上下文窗口","relation":"衡量单位","object":"token"},{"subject":"Agent","relation":"执行方式","object":"多轮执行"}]

注意：上面的示例只用于说明格式，只有当下面文本真实支持这些事实时才能抽取。

【文本内容】
{text}

【JSON 输出】
"""


_TECH_ENTITY_CANONICAL = {
    "agent": "Agent",
    "neo4j": "Neo4j",
    "rag": "RAG",
    "llm": "LLM",
    "api": "API",
    "token": "token",
}


def _clean_text_field(value: object) -> str:
    """清理模型输出字段中的多余空白、引号和句末标点。"""
    if not isinstance(value, str):
        return ""

    text = re.sub(r"\s+", " ", value.strip())
    text = text.strip('“”"\'` ')
    text = text.rstrip("。；;，,")
    return text.strip()


def _normalize_entity(value: object) -> str:
    """对实体做保守归一化，不做语义猜测。"""
    entity = _clean_text_field(value)
    if not entity:
        return ""

    # 英文技术词仅在“整个实体完全等于该术语”时做大小写规范化。
    canonical = _TECH_ENTITY_CANONICAL.get(entity.casefold())
    if canonical:
        return canonical

    # 统一中英文之间的多余空格，例如 "Agent 工程" 保持可读但不重复空白。
    entity = re.sub(r"\s+", " ", entity)
    return entity


def _normalize_relation(value: object) -> str:
    relation = _clean_text_field(value)
    if not relation:
        return ""

    # 关系名不应该是一整句；只做长度保护，不替模型发明新关系。
    if len(relation) > 20:
        return ""
    return relation


def _extract_json_array(raw_output: str) -> list:
    """兼容模型偶尔输出 ```json ... ``` 或前后少量文字。"""
    cleaned = (raw_output or "").strip()
    if not cleaned:
        return []

    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _sanitize_triples(items: list) -> list[dict]:
    """字段校验 + 清洗 + 去重，防止脏数据写入 Neo4j。"""
    results: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        subject = _normalize_entity(item.get("subject"))
        relation = _normalize_relation(item.get("relation"))
        obj = _normalize_entity(item.get("object"))

        if not subject or not relation or not obj:
            continue
        if subject.casefold() == obj.casefold():
            continue

        # 防止整段句子被误当成节点。这里只做宽松上限，中文定义短语仍可保留。
        if len(subject) > 80 or len(obj) > 100:
            continue

        key = (subject.casefold(), relation.casefold(), obj.casefold())
        if key in seen:
            continue

        seen.add(key)
        results.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
        })

        if len(results) >= 10:
            break

    return results


def extract_triples(text: str) -> list[dict]:
    """从一段文本抽取并清洗三元组。失败时返回空列表，不中断文档入库。"""
    if not text or not text.strip():
        return []

    prompt = EXTRACTION_PROMPT.replace("{text}", text)

    try:
        response = _llm_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_output = response.choices[0].message.content or ""
        items = _extract_json_array(raw_output)
        return _sanitize_triples(items)

    except Exception as exc:
        print(f"三元组抽取失败: {exc}")
        return []
