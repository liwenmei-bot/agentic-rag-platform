"""
问答接口。

本版在原有 Router / Context Judge / Query Rewrite SSE 的基础上，
显式透传 Planner / Actor / Reflector 事件，用于前端展示完整 Agent 闭环。
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services import session_service
from app.services.memory_service import generation_context, resolve_question
from app.services.rag_service import (
    answer_question,
    basic_answer_question,
    stream_answer,
    stream_basic_answer,
)


router = APIRouter()
logger = logging.getLogger(__name__)


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
    context_sufficient: bool | None = None

    # Planner / Actor / Reflector trace
    planner: dict | None = None
    actor_trace: list[dict] = Field(default_factory=list)
    reflection: dict | None = None
    reflection_history: list[dict] = Field(default_factory=list)


class StreamChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)


def _validate_stream_request(request: StreamChatRequest) -> list[dict]:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="问题不能为空")
    if not session_service.session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return session_service.get_recent_turns(request.session_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非流式问答接口，返回最终答案和完整 Agent Trace。"""
    result = answer_question(request.question)

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        rewrite_attempted=result.get("rewrite_attempted", False),
        rewrite_candidate=result.get("rewrite_candidate"),
        rewritten_query=result.get("rewritten_query"),
        context_sufficient=result.get("context_sufficient"),
        planner=result.get("planner"),
        actor_trace=result.get("actor_trace", []),
        reflection=result.get("reflection"),
        reflection_history=result.get("reflection_history", []),
    )


def _sse_format(event_type: str, data) -> str:
    """把数据包装成 SSE（Server-Sent Events）格式。"""
    payload = json.dumps(
        {"type": event_type, "data": data},
        ensure_ascii=False,
    )
    return f"data: {payload}\n\n"


@router.post("/chat/basic", response_model=ChatResponse)
async def basic_chat(request: ChatRequest):
    """Basic RAG 非流式接口：固定一次 Chroma 检索，不进入 Agent 编排。"""
    result = basic_answer_question(request.question)
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        route=result.get("route"),
        route_reason=result.get("route_reason"),
    )


@router.post("/chat/basic/stream")
async def basic_chat_stream(request: StreamChatRequest):
    """Basic RAG 流式接口：Question -> Chroma -> LLM stream。"""
    history = _validate_stream_request(request)

    def event_generator():
        full_answer = ""
        sources_data = []

        try:
            yield _sse_format("stage", {"stage": "memory", "status": "running"})
            resolved = resolve_question(request.question, history)
            yield _sse_format("memory", {"resolved_question": resolved, "turns": len(history)})
            prompt = generation_context(request.question, resolved, history)
            for event in stream_basic_answer(resolved, prompt_question=prompt):
                event_type = event["type"]
                event_data = event["data"]
                if event_type == "sources":
                    sources_data = event_data or []
                elif event_type == "content":
                    full_answer += event_data or ""
                yield _sse_format(event_type, event_data)

            if not full_answer.strip():
                raise RuntimeError("Basic RAG returned an empty answer")
            session_service.add_exchange(
                request.session_id, request.question, full_answer,
                sources=json.dumps(sources_data, ensure_ascii=False),
            )
            yield _sse_format("done", {"mode": "basic_rag", "status": "ok"})
        except Exception:
            logger.exception("Basic RAG stream failed")
            yield _sse_format("error", {"message": "回答生成失败，请稍后重试。"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/stream")
async def chat_stream(request: StreamChatRequest):
    """
    流式问答接口。

    rag_service.stream_answer() 可能产生：
    - planner
    - actor
    - reflector
    - retrieval_info
    - sources
    - content
    """
    history = _validate_stream_request(request)

    def event_generator():
        full_answer = ""
        sources_data = []
        retrieval_info_data = None

        planner_data = None
        actor_trace_data = []
        reflection_history_data = []

        try:
            yield _sse_format("stage", {"stage": "memory", "status": "running"})
            resolved = resolve_question(request.question, history)
            yield _sse_format("memory", {"resolved_question": resolved, "turns": len(history)})
            prompt = generation_context(request.question, resolved, history)
            for event in stream_answer(resolved, prompt_question=prompt):
                event_type = event["type"]
                event_data = event["data"]
                if event_type == "planner":
                    planner_data = event_data
                elif event_type == "actor":
                    actor_trace_data.append(event_data)
                elif event_type == "reflector":
                    reflection_history_data.append(event_data)
                elif event_type == "retrieval_info":
                    retrieval_info_data = event_data
                elif event_type == "sources":
                    sources_data = event_data or []
                elif event_type == "content":
                    full_answer += event_data or ""
                yield _sse_format(event_type, event_data)

            if not full_answer.strip():
                raise RuntimeError("Agentic RAG returned an empty answer")
            session_service.add_exchange(
                request.session_id, request.question, full_answer,
                sources=json.dumps(sources_data, ensure_ascii=False),
            )
            yield _sse_format("done", {
                "status": "ok",
                "retrieval_info": retrieval_info_data,
                "planner": planner_data,
                "actor_trace": actor_trace_data,
                "reflection_history": reflection_history_data,
            })
        except Exception:
            logger.exception("Agentic RAG stream failed")
            yield _sse_format("error", {"message": "回答生成失败，请稍后重试。"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
