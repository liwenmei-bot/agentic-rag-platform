"""
Agent 对话接口。

跟第二阶段的 /chat/stream 结构类似，同样用 SSE 推送，
但这里额外会推送 tool_call / tool_result / file 这几类事件，
前端据此可以显示"正在调用xx工具"这样的过程提示，而不只是最终答案。
"""
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services import session_service
from app.services.agent_service import run_agent

router = APIRouter()

WORKSPACE_ROOT = Path(settings.upload_dir).parent / "workspace"


class AgentChatRequest(BaseModel):
    session_id: str
    question: str


def _sse_format(event_type: str, data) -> str:
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


@router.post("/agent/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    session_service.add_message(request.session_id, role="user", content=request.question)
    workspace_dir = WORKSPACE_ROOT / request.session_id

    def event_generator():
        full_answer = ""
        generated_files = []

        for event in run_agent(request.question, workspace_dir):
            if event["type"] == "content":
                full_answer += event["data"]
            elif event["type"] == "file":
                generated_files.append(event["data"])
            yield _sse_format(event["type"], event["data"])

        session_service.add_message(
            request.session_id,
            role="assistant",
            content=full_answer,
            sources=json.dumps({"files": generated_files}, ensure_ascii=False),
        )
        yield _sse_format("done", None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
