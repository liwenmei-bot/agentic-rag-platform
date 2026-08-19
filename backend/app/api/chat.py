"""
问答接口。

第一阶段先做非流式的简单版本（一次性返回完整答案），
流式输出（打字机效果）放在第二阶段做前端时再升级成 SSE，
这样每个阶段的改动范围更清晰，方便你理解"从能用到好用"的演进过程。
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.rag_service import answer_question

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    filename: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = answer_question(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])
