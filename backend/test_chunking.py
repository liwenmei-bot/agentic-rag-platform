"""
简单测试脚本，验证切块逻辑（不需要 API Key，不需要联网，可以直接跑）。
运行方式：python test_chunking.py
"""
from app.utils.chunking import chunk_text, split_into_sentences

sample_text = (
    "语析是一个开源的智能知识库与知识图谱智能体开发平台。"
    "它把 RAG 检索、知识库内知识图谱与多智能体编排整合进统一的工作台。"
    "管理员可以配置知识库、模型与权限。"
    "用户在类 ChatGPT 的界面中与智能体对话，获得带引用来源的回答。"
    "这个项目基于 Vue 3、FastAPI 和 LangGraph 构建。"
)

if __name__ == "__main__":
    print("=== 分句测试 ===")
    sentences = split_into_sentences(sample_text)
    for i, s in enumerate(sentences, 1):
        print(f"{i}. {s}")

    print("\n=== 切块测试 (chunk_size=50, overlap=10) ===")
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10)
    for i, c in enumerate(chunks, 1):
        print(f"--- Chunk {i} (长度 {len(c)}) ---")
        print(c)
        print()
