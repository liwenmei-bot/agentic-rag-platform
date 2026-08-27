"""
全局配置管理。
所有环境变量在这里统一读取，其他模块只从这里 import settings，不要到处写 os.getenv。
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    llm_api_key: str = "your_api_key_here"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model_name: str = "deepseek-chat"

    # Embedding
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    # 存储路径
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    # 切块与检索参数
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # Neo4j 知识图谱
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_neo4j_password_here"

    # Agent 工具：联网搜索（可选，不配置时该工具会返回提示而不是报错）
    serper_api_key: str = "your_serper_api_key_here"


# 全局单例，其他地方 from app.core.config import settings 直接用
settings = Settings()
