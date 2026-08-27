"""
FastAPI 应用入口。

运行方式：
    cd backend
    uvicorn app.main:app --reload --port 8000

跑起来后访问 http://localhost:8000/docs 可以看到自动生成的交互式 API 文档，
不需要额外写前端就能直接在浏览器里测试上传文档和问答接口。
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import upload, chat, sessions, graph, agent
from app.core.config import settings
from app.services.session_service import init_db

app = FastAPI(title="agentic-rag-platform - Phase 4: Agent Tool Calling")

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
app.include_router(agent.router, prefix="/api", tags=["agent"])

# Agent 生成的文件（比如 generate_report 工具产出的 Markdown 报告）存在这个目录，
# 挂载成静态文件服务后，前端可以直接用 /files/{session_id}/{filename} 这样的 URL 下载
WORKSPACE_ROOT = Path(settings.upload_dir).parent / "workspace"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(WORKSPACE_ROOT)), name="files")


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/health")
async def health_check():
    return {"status": "ok"}
