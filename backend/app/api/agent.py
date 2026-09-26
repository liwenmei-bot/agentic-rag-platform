"""
Agent 对话接口。

跟第二阶段的 /chat/stream 结构类似，同样用 SSE 推送，
但这里额外会推送 tool_call / tool_result / file 这几类事件，
前端据此可以显示"正在调用xx工具"这样的过程提示，而不只是最终答案。
"""
import json
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services import session_service
from app.services.agent_service import run_agent

router = APIRouter()
logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(settings.upload_dir).parent / "workspace"


class AgentChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)


def _sse_format(event_type: str, data) -> str:
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


@router.post("/agent/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="问题不能为空")
    try:
        UUID(request.session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="无效的会话 ID") from None
    if not session_service.session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    history = session_service.get_recent_turns(request.session_id)
    workspace_dir = WORKSPACE_ROOT / request.session_id

    def event_generator():
        full_answer = ""
        generated_files = []

        try:
            for event in run_agent(request.question, workspace_dir, history=history):
                if event["type"] == "content":
                    full_answer += event["data"]
                elif event["type"] == "file":
                    generated_files.append(event["data"])
                yield _sse_format(event["type"], event["data"])
            if not full_answer.strip():
                raise RuntimeError("Tool Agent returned an empty answer")
            session_service.add_exchange(
                request.session_id, request.question, full_answer,
                sources=json.dumps({"files": generated_files}, ensure_ascii=False),
            )
            yield _sse_format("done", {"status": "ok"})
        except Exception:
            logger.exception("Tool Agent stream failed")
            yield _sse_format("error", {"message": "工具 Agent 运行失败，请稍后重试。"})

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
