"""
向量库服务：封装 Chroma 的存储、更新与检索逻辑。

本版新增：
1. 同名文件重新上传时，先删除该 filename 的旧 chunks，避免重复入库。
2. search() 对 filename + chunk_index 做去重，避免 Top-K 被重复 chunk 占满。
3. 查询时多取一些候选结果，再从中挑选去重后的 Top-K。
"""

from functools import lru_cache

import chromadb

from app.core.config import settings
from app.services.embedding_service import embed_query, embed_texts


COLLECTION_NAME = "documents"


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def delete_chunks_by_filename(filename: str) -> int:
    """
    删除向量库中指定文件名对应的全部 chunks。

    用途：
    同名文档重新上传时，把旧版本先移除，避免因为每次生成新的 doc_id
    而在 Chroma 中不断累积重复内容。
    """
    if not filename:
        return 0

    collection = get_collection()

    existing = collection.get(
        where={"filename": filename},
    )
    ids = existing.get("ids") or []

    if not ids:
        return 0

    collection.delete(ids=ids)
    return len(ids)


def add_chunks(doc_id: str, filename: str, chunks: list[str]) -> None:
    """
    把一份文档切好的 chunks 存入向量库。

    同名文件采用“替换”语义：
    - 先完成 embedding；
    - 再删除该 filename 的旧 chunks；
    - 最后写入当前版本。

    这样用户反复上传同一个文件时，不会得到多份相同 chunk。
    """
    if not chunks:
        return

    collection = get_collection()

    # 先算 embedding；如果模型阶段失败，不会提前删除旧数据。
    embeddings = embed_texts(chunks)

    # 同名文档重新上传视为更新，而不是新增一份副本。
    delete_chunks_by_filename(filename)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": filename,
            "doc_id": doc_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def _dedupe_key(document: str, metadata: dict) -> tuple:
    """
    优先用 filename + chunk_index 判断同一逻辑 chunk。
    对历史旧数据如果缺少 chunk_index，则回退到 filename + 文本内容。
    """
    filename = metadata.get("filename")
    chunk_index = metadata.get("chunk_index")

    if chunk_index is not None:
        return ("chunk", filename, chunk_index)

    return ("content", filename, (document or "").strip())


def search(query: str, top_k: int = 4) -> list[dict]:
    """
    检索与 query 最相关的文本块。

    为了避免历史重复数据占满 Top-K：
    1. 先取 top_k * 4 个候选；
    2. 按 filename + chunk_index 去重；
    3. 保留排序最靠前的 top_k 个唯一结果。
    """
    if not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    total_count = collection.count()

    if total_count <= 0:
        return []

    query_embedding = embed_query(query)

    candidate_count = min(
        total_count,
        max(top_k * 4, top_k),
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits = []
    seen = set()

    for doc, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        key = _dedupe_key(doc, meta)

        if key in seen:
            continue

        seen.add(key)

        hits.append(
            {
                "content": doc,
                "filename": meta.get("filename"),
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                # 保留项目原有评分方式，避免这一轮同时改变 Context Judge 的阈值语义。
                # 注意：该值是检索排序分数，不应当作概率解释。
                "score": 1 - float(distance),
            }
        )

        if len(hits) >= top_k:
            break

    return hits
