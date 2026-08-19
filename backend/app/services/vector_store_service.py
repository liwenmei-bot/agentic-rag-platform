"""
向量库服务：封装 Chroma 的存储与检索逻辑。

用 Chroma 是因为它零配置、本地文件持久化，适合个人项目起步。
后期如果数据量大了、需要更高并发，可以把这一层换成 Milvus，
上层调用代码（rag_service.py）基本不用改，这就是为什么要单独封装一层的原因。
"""
from functools import lru_cache

import chromadb

from app.core.config import settings
from app.services.embedding_service import embed_texts, embed_query

COLLECTION_NAME = "documents"


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunks(doc_id: str, filename: str, chunks: list[str]) -> None:
    """把一份文档切好的 chunks 存入向量库。"""
    if not chunks:
        return

    collection = get_collection()
    embeddings = embed_texts(chunks)

    # 每个 chunk 一个唯一 id，方便后续按文档删除/更新
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def search(query: str, top_k: int = 4) -> list[dict]:
    """检索与 query 最相关的 top_k 个文本块，返回内容 + 来源信息。"""
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        hits.append({
            "content": doc,
            "filename": meta.get("filename"),
            "chunk_index": meta.get("chunk_index"),
            "score": 1 - distance,  # Chroma 默认返回距离，转成相似度分数更直观
        })

    return hits
