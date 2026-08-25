"""
会话管理接口。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import session_service

router = APIRouter()


class CreateSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    session = session_service.create_session()
    return session


@router.get("/sessions")
async def list_sessions():
    return session_service.list_sessions()


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    return session_service.get_session_messages(session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    session_service.delete_session(session_id)
    return {"message": "会话已删除"}
