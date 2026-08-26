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

from app.api import upload, chat, sessions, graph
from app.services.session_service import init_db

app = FastAPI(title="agentic-rag-platform - Phase 3: Knowledge Graph")

# 开发阶段允许所有来源跨域，方便本地前端调试；正式部署时应改成具体域名白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(graph.router, prefix="/api", tags=["graph"])


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/health")
async def health_check():
    return {"status": "ok"}
