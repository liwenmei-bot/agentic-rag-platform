"""
RAG 核心逻辑：
1. 文档入库：解析 -> 切块 -> 存入向量库 -> 抽取三元组存入知识图谱
2. 问答：向量检索 + 图谱增强检索 -> 拼接 prompt -> 调用 LLM -> 返回带来源的回答
"""
import uuid

from openai import OpenAI

from app.core.config import settings
from app.services import graph_service
from app.services.extraction_service import extract_triples
from app.services.vector_store_service import add_chunks, search
from app.utils.chunking import chunk_text
from app.utils.parsing import parse_document

_llm_client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

# Prompt 设计要点（对应路线图里提到的"防止模型瞎编"）：
# 1. 明确要求"只根据资料回答"
# 2. 明确要求"找不到就说找不到"，不许编造
# 3. 要求标注引用来源，方便用户核实
# 4. 第三阶段新增：允许模型参考"知识图谱关联信息"做更深层的关联回答
SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请遵守以下规则：
1. 只根据下面提供的【参考资料】和【知识图谱关联信息】回答问题，不要使用你自己的知识编造内容。
2. 如果提供的信息都不足以回答问题，必须明确说"根据现有资料无法回答这个问题"，不要编造答案。
3. 回答时，在相关内容后面用【来源：文件名】的格式标注引用出处；如果某个结论是从知识图谱关系推理得出的，标注【来源：知识图谱】。
4. 回答要简洁准确，不要重复参考资料的原文，用自己的话组织答案。
"""


def ingest_document(file_path: str, filename: str) -> dict:
    """
    把一份文档解析、切块、存入向量库，并抽取三元组存入知识图谱。

    三元组抽取是逐个 chunk 调用 LLM 完成的，所以文档越大、chunk 越多，
    这一步耗时会越长（每个 chunk 都要额外调一次 LLM）。
    """
    text = parse_document(file_path)
    if not text.strip():
        raise ValueError("文档解析后内容为空，请检查文件是否损坏或是扫描版 PDF（暂不支持 OCR）")

    chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    doc_id = str(uuid.uuid4())
    add_chunks(doc_id=doc_id, filename=filename, chunks=chunks)

    triple_count = 0
    for chunk in chunks:
        triples = extract_triples(chunk)
        if triples:
            triple_count += graph_service.add_triples(triples, source_filename=filename)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "triple_count": triple_count,
    }


def _build_graph_context(question: str) -> str:
    """
    图谱增强检索的核心逻辑：
    1. 找出图谱里有哪些实体名称是这次提问里提到的
    2. 查询这些实体在图谱中的邻居关系
    3. 把查到的关系拼成一段文字，作为额外的参考信息喂给 LLM

    这一步是"图谱增强"相较"纯向量检索"多出来的能力——
    向量检索找的是"语义相似的文字片段"，图谱检索找的是"明确的实体关系"，
    两者互补：向量检索适合"文中直接提到的内容"，图谱检索适合
    "需要跨文档、跨段落关联多个实体才能回答"的问题。
    """
    all_entities = graph_service.list_all_entity_names()
    mentioned = [name for name in all_entities if name and name in question]

    if not mentioned:
        return ""

    relations = graph_service.find_related_entities(mentioned)
    if not relations:
        return ""

    lines = [f"{r['source']} -[{r['relation']}]-> {r['target']}" for r in relations]
    return "\n".join(lines)


def _build_prompt(question: str, hits: list[dict], graph_context: str) -> str:
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(f"[资料{i}] 来源：{hit['filename']}\n{hit['content']}")
    context_text = "\n\n".join(context_parts) if context_parts else "（无相关文档片段）"

    graph_section = f"\n\n【知识图谱关联信息】\n{graph_context}" if graph_context else ""

    return f"""【参考资料】
{context_text}{graph_section}

【用户问题】
{question}
"""


def answer_question(question: str) -> dict:
    """检索相关片段（向量 + 图谱），调用 LLM 生成带引用的回答。"""
    hits = search(question, top_k=settings.top_k)
    graph_context = _build_graph_context(question)

    if not hits and not graph_context:
        return {
            "answer": "知识库中还没有任何文档，请先上传文档再提问。",
            "sources": [],
        }

    user_prompt = _build_prompt(question, hits, graph_context)

    response = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    sources = [{"filename": hit["filename"], "score": round(hit["score"], 3)} for hit in hits]

    return {"answer": answer, "sources": sources}


def stream_answer(question: str):
    """流式版本：一边生成一边把文字片段 yield 出去，前端就能做打字机效果。"""
    hits = search(question, top_k=settings.top_k)
    graph_context = _build_graph_context(question)

    if not hits and not graph_context:
        yield {"type": "sources", "data": []}
        yield {"type": "content", "data": "知识库中还没有任何文档，请先上传文档再提问。"}
        return

    user_prompt = _build_prompt(question, hits, graph_context)
    sources = [{"filename": hit["filename"], "score": round(hit["score"], 3)} for hit in hits]
    yield {"type": "sources", "data": sources}

    stream = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "content", "data": delta}
