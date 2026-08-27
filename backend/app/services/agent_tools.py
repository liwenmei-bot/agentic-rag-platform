"""
Agent 工具集。

每个工具需要两部分：
1. TOOL_SCHEMAS 里的 JSON Schema 描述——喂给 LLM，让它知道"有哪些工具可用、
   每个工具需要什么参数"，模型会根据这个描述自己判断要不要调用、怎么调用
2. execute_tool() 里的实际执行逻辑——模型决定调用某个工具后，
   代码真正去执行这个工具，把结果返回给模型

这是 Function Calling / Tool Use 的标准模式：模型不直接"做事"，
而是"说它想用哪个工具、传什么参数"，代码负责真正执行，再把结果喂回去给模型。
"""
from pathlib import Path

import requests

from app.core.config import settings
from app.services.rag_service import _build_graph_context
from app.services.vector_store_service import search as vector_search

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在用户已上传的知识库文档中检索相关信息，包括向量检索和知识图谱关联信息。当问题可能与已上传文档相关时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题或关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索最新信息。当问题涉及知识库之外的内容、需要最新资讯、或用户明确要求联网查找时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "把内容整理成一份 Markdown 格式的报告文件，生成后用户可以下载。当用户要求生成报告、总结文档、或产出一份可下载的文件时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "报告标题，会作为文件名的一部分"},
                    "content": {"type": "string", "description": "报告正文内容，使用 Markdown 格式"},
                },
                "required": ["title", "content"],
            },
        },
    },
]


def _tool_search_knowledge_base(query: str) -> str:
    hits = vector_search(query, top_k=settings.top_k)
    graph_context = _build_graph_context(query)

    if not hits and not graph_context:
        return "知识库中没有找到相关信息。"

    parts = []
    if hits:
        parts.append("向量检索结果：")
        for hit in hits:
            parts.append(f"- [{hit['filename']}] {hit['content'][:200]}")
    if graph_context:
        parts.append("\n知识图谱关联信息：")
        parts.append(graph_context)

    return "\n".join(parts)


def _tool_web_search(query: str) -> str:
    """
    联网搜索，用的是 Serper.dev 的 Google 搜索 API（有免费额度，需要单独申请 key）。
    如果没配置 key，返回明确的提示而不是报错崩溃——这样 Agent 会知道这个工具暂时不可用，
    转而依赖其他信息源或者如实告诉用户。
    """
    if not settings.serper_api_key or settings.serper_api_key == "your_serper_api_key_here":
        return "联网搜索功能未配置（缺少 SERPER_API_KEY），暂时无法联网搜索，请基于已有信息回答，或提示用户配置该功能。"

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", [])[:5]:
            results.append(f"- {item.get('title', '')}: {item.get('snippet', '')} ({item.get('link', '')})")

        return "\n".join(results) if results else "联网搜索没有找到相关结果。"

    except Exception as e:
        return f"联网搜索失败：{str(e)}"


def _tool_generate_report(title: str, content: str, workspace_dir: Path) -> dict:
    """把内容写成 Markdown 文件，存到这个会话专属的 workspace 目录下。"""
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 简单清洗文件名，避免标题里的特殊字符导致文件系统报错
    safe_title = "".join(c for c in title if c.isalnum() or c in "-_ ")[:50].strip() or "report"
    filename = f"{safe_title}.md"
    file_path = workspace_dir / filename

    file_path.write_text(content, encoding="utf-8")

    return {"filename": filename, "path": str(file_path)}


def execute_tool(tool_name: str, arguments: dict, workspace_dir: Path):
    """
    统一的工具执行入口。

    返回 (给模型看的文字结果, 给前端看的附加信息)：
    - 第一个值会作为 "tool" 角色的消息喂回给 LLM，让它继续推理/生成最终回答
    - 第二个值只有 generate_report 会用到（文件信息），用来告诉前端"生成了一个文件，可以下载"
    """
    if tool_name == "search_knowledge_base":
        result = _tool_search_knowledge_base(arguments.get("query", ""))
        return result, None

    elif tool_name == "web_search":
        result = _tool_web_search(arguments.get("query", ""))
        return result, None

    elif tool_name == "generate_report":
        file_info = _tool_generate_report(
            arguments.get("title", "report"),
            arguments.get("content", ""),
            workspace_dir,
        )
        return f"报告已生成：{file_info['filename']}", file_info

    else:
        return f"未知工具：{tool_name}", None
