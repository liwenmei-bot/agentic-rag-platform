"""
Embedding 服务：把文本转成向量。

用 sentence-transformers 在本地加载 bge-small-zh，不依赖外部 API，
免费、无网络延迟，适合个人项目起步阶段。
模型第一次运行时会自动从 HuggingFace 下载（几十MB，几分钟内完成）。
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """懒加载 + 缓存模型实例，避免每次调用都重新加载模型（很耗时）。"""
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量列表。"""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """把单条查询文本转成向量（检索时用）。"""
    return embed_texts([query])[0]
