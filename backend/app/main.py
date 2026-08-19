"""
FastAPI 应用入口。

运行方式：
    cd backend
    uvicorn app.main:app --reload --port 8000

跑起来后访问 http://localhost:8000/docs 可以看到自动生成的交互式 API 文档，
不需要额外写前端就能直接在浏览器里测试上传文档和问答接口。
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, chat

app = FastAPI(title="My Agent Platform - Phase 1: Basic RAG")

# 开发阶段允许所有来源跨域，方便本地前端调试；正式部署时应改成具体域名白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
