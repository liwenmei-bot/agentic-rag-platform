"""
RAG 核心逻辑：
1. 文档入库：解析 -> 切块 -> 存入向量库
2. 问答：检索相关片段 -> 拼接 prompt -> 调用 LLM -> 返回带来源的回答
"""
import uuid

from openai import OpenAI

from app.core.config import settings
from app.services.vector_store_service import add_chunks, search
from app.utils.chunking import chunk_text
from app.utils.parsing import parse_document

_llm_client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

# Prompt 设计要点（对应路线图里提到的"防止模型瞎编"）：
# 1. 明确要求"只根据资料回答"
# 2. 明确要求"找不到就说找不到"，不许编造
# 3. 要求标注引用来源，方便用户核实
SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请遵守以下规则：
1. 只根据下面提供的【参考资料】回答问题，不要使用你自己的知识编造内容。
2. 如果参考资料中没有足够信息回答问题，必须明确说"根据现有资料无法回答这个问题"，不要编造答案。
3. 回答时，在相关内容后面用【来源：文件名】的格式标注引用出处。
4. 回答要简洁准确，不要重复参考资料的原文，用自己的话组织答案。
"""


def ingest_document(file_path: str, filename: str) -> dict:
    """把一份文档解析、切块、存入向量库。返回处理结果统计信息。"""
    text = parse_document(file_path)
    if not text.strip():
        raise ValueError("文档解析后内容为空，请检查文件是否损坏或是扫描版 PDF（暂不支持 OCR）")

    chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    doc_id = str(uuid.uuid4())
    add_chunks(doc_id=doc_id, filename=filename, chunks=chunks)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
    }


def answer_question(question: str) -> dict:
    """检索相关片段，调用 LLM 生成带引用的回答。"""
    hits = search(question, top_k=settings.top_k)

    if not hits:
        return {
            "answer": "知识库中还没有任何文档，请先上传文档再提问。",
            "sources": [],
        }

    # 把检索到的片段拼成参考资料文本
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(f"[资料{i}] 来源：{hit['filename']}\n{hit['content']}")
    context_text = "\n\n".join(context_parts)

    user_prompt = f"""【参考资料】
{context_text}

【用户问题】
{question}
"""

    response = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # 低温度，减少模型自由发挥，问答场景要准确不要"创意"
    )

    answer = response.choices[0].message.content

    sources = [
        {"filename": hit["filename"], "score": round(hit["score"], 3)}
        for hit in hits
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
