"""
问答接口。

第一阶段的非流式接口（/chat）保留，方便用 /docs 页面快速测试。
第二阶段新增流式接口（/chat/stream），前端对话界面用这个，实现打字机效果，
并且会把问答记录存进对应的会话（session），支持多轮对话历史。

本版新增：
1. /api/chat 返回 rewrite_attempted / rewrite_candidate / rewritten_query，便于观察 Query Rewrite 全过程。
2. sources 增加 chunk_index，与新版 rag_service.py 的精确段落引用保持一致。
3. /api/chat/stream 透传 retrieval_info 事件，让前端可以看到查询是否尝试改写、候选查询以及是否最终采用。
"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import session_service
from app.services.rag_service import answer_question, stream_answer


router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    filename: str
    score: float
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    route: str | None = None
    route_reason: str | None = None
    rewrite_attempted: bool = False
    rewrite_candidate: str | None = None
    rewritten_query: str | None = None


class StreamChatRequest(BaseModel):
    session_id: str
    question: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    非流式问答接口。

    rewritten_query:
    - None：第一次检索已足够，没有触发查询改写
    - 字符串：第一次检索不充分，系统执行了 Query Rewrite
    """
    result = answer_question(request.question)

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        rewrite_attempted=result.get("rewrite_attempted", False),
        rewrite_candidate=result.get("rewrite_candidate"),
        rewritten_query=result.get("rewritten_query"),
    )


def _sse_format(event_type: str, data) -> str:
    """
    把数据包装成 SSE（Server-Sent Events）协议要求的格式。
    """
    payload = json.dumps(
        {"type": event_type, "data": data},
        ensure_ascii=False,
    )
    return f"data: {payload}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: StreamChatRequest):
    """
    流式问答接口。

    新版 rag_service.stream_answer() 会产生：
    - retrieval_info
    - sources
    - content
    """
    session_service.add_message(
        request.session_id,
        role="user",
        content=request.question,
    )

    def event_generator():
        full_answer = ""
        sources_data = []
        retrieval_info_data = None

        for event in stream_answer(request.question):
            event_type = event.get("type")
            event_data = event.get("data")

            if event_type == "retrieval_info":
                retrieval_info_data = event_data
                yield _sse_format("retrieval_info", retrieval_info_data)

            elif event_type == "sources":
                sources_data = event_data or []
                yield _sse_format("sources", sources_data)

            elif event_type == "content":
                text = event_data or ""
                full_answer += text
                yield _sse_format("content", text)

        session_service.add_message(
            request.session_id,
            role="assistant",
            content=full_answer,
            sources=json.dumps(sources_data, ensure_ascii=False),
        )

        yield _sse_format(
            "done",
            {"retrieval_info": retrieval_info_data},
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
