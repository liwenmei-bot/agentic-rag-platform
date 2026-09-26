# 后端开发指南

本目录是 FastAPI 后端。本地运行需要 Python 3.11、Neo4j、LLM API，以及首次加载 Embedding 模型时的下载能力。完整架构和 SSE 时序见 [项目实现文档](../docs/pipeline-memory-evaluation.md)。

## Windows PowerShell

```powershell
cd backend
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env 中的 LLM_API_KEY、NEO4J_URI、NEO4J_PASSWORD
uvicorn app.main:app --reload --port 8000
```

## macOS / Linux

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Neo4j 本地示例：

```bash
docker run -d --name neo4j-agentic-rag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/你的密码 neo4j:5.24
```

根目录 `docker compose up --build` 会代为启动数据库和前后端，不需要额外启动此容器。接口文档：`http://localhost:8000/docs`；健康检查：`http://localhost:8000/health`。

## 关键接口

| 方法与路径 | 作用 |
| --- | --- |
| `POST /api/upload` | PDF/DOCX/TXT 上传、解析、向量化与图谱抽取 |
| `POST /api/sessions`、`GET /api/sessions` | 新建、列出会话 |
| `GET /api/sessions/{id}/messages`、`DELETE /api/sessions/{id}` | 查看、删除历史 |
| `POST /api/chat/basic/stream` | Basic RAG 流式问答 |
| `POST /api/chat/stream` | 四路 Router + 真实管线 SSE |
| `POST /api/agent/chat/stream` | Tool Agent 流式问答 |
| `GET /api/graph` | 图谱可视化数据 |

流式请求体是 `{"session_id":"<POST /api/sessions 返回的 id>","question":"..."}`。问答成功后将用户与助手消息作为完整轮次写入 SQLite；服务报错或断连不保存半轮。历史按最近 8 条、共最多 4000 字读取，单条最多 1000 字；只有短追问才尝试调用 LLM 消歧，失败返回原问题。独立问题仍可参考有界对话上下文，事实回答须基于检索证据。

## 测试和评估

```bash
python -m pytest -q tests/
python -m evaluation.test_agent_orchestrator
python -m evaluation.test_par_retry_loop
python -m evaluation.evaluate_answers --mode live --limit 20
```

前三项为离线检查，最后一项调用实际服务和 LLM，必须先上传与评测集一致的资料。评估输出在 `evaluation/results/answers/`。`--judge` 会额外调用 LLM Judge；本项目的关键词覆盖率只是诊断指标，不当作回答准确率。不要用开发集、回归集替代冻结后一次性独立 Holdout 的成绩。

## 本地运行边界

项目目前没有登录/权限控制；`/files` 可在本地下载该工作区生成的报告，不能直接对公网开放。Chroma/Neo4j 和模型的实际行为依赖运行环境；无 API Key 不能完成真实回答评测。详细见 [项目实现文档](../docs/pipeline-memory-evaluation.md)。
