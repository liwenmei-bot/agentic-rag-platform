"""
问答接口。

第一阶段的非流式接口（/chat）保留，方便用 /docs 页面快速测试。
第二阶段新增流式接口（/chat/stream），前端对话界面用这个，实现打字机效果，
并且会把问答记录存进对应的会话（session），支持多轮对话历史。
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class StreamChatRequest(BaseModel):
    session_id: str
    question: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = answer_question(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


def _sse_format(event_type: str, data) -> str:
    """
    把数据包装成 SSE（Server-Sent Events）协议要求的格式：
    每条消息以 "data: " 开头，以两个换行结尾。
    我们额外用 JSON 包一层 type 字段，方便前端区分"这是来源信息"还是"这是正文内容"。
    """
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: StreamChatRequest):
    # 先把用户的问题存进历史记录
    session_service.add_message(request.session_id, role="user", content=request.question)

    def event_generator():
        full_answer = ""
        sources_data = []

        for event in stream_answer(request.question):
            if event["type"] == "sources":
                sources_data = event["data"]
                yield _sse_format("sources", sources_data)
            elif event["type"] == "content":
                full_answer += event["data"]
                yield _sse_format("content", event["data"])

        # 流式生成结束后，把完整回答也存进历史记录，供下次打开会话时回显
        session_service.add_message(
            request.session_id,
            role="assistant",
            content=full_answer,
            sources=json.dumps(sources_data, ensure_ascii=False),
        )
        yield _sse_format("done", None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
