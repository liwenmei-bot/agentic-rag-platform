# agentic-rag-platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen.svg)

一个融合 RAG 检索、知识图谱与 Agent 工具调用的智能知识平台（开发中）。
参考 [xerrors/Yuxi](https://github.com/xerrors/Yuxi) 等开源项目的架构思路，独立设计并实现。

技术栈：FastAPI · Chroma · sentence-transformers · DeepSeek API（后续阶段将加入 Vue 3 前端、Neo4j 知识图谱、Agent 工具调用）

## 开发进度

- [x] **第一阶段**：基础 RAG 问答（文档解析 → 切块 → 向量检索 → 带引用回答）
- [ ] 第二阶段：Vue 3 对话式前端 + 流式输出
- [ ] 第三阶段：知识图谱融合（Neo4j + 图谱增强检索）
- [ ] 第四阶段：Agent 工具调用（联网搜索、代码执行、文件生成）
- [ ] 第五阶段：Docker 一键部署

## 第一阶段：基础 RAG 问答后端

对应开发路线图「第一阶段」的骨架代码：上传文档 → 解析 → 切块 → 向量化 → 检索 → 生成带引用的回答。

## 目录结构

```
backend/
├── app/
│   ├── main.py                     # FastAPI 入口
│   ├── api/
│   │   ├── upload.py                # 上传接口
│   │   └── chat.py                  # 问答接口
│   ├── core/
│   │   └── config.py                # 统一配置管理
│   ├── services/
│   │   ├── embedding_service.py     # 向量化封装
│   │   ├── vector_store_service.py  # Chroma 存取封装
│   │   └── rag_service.py           # 核心 RAG 逻辑（入库 + 问答）
│   └── utils/
│       ├── parsing.py               # 文档解析（pdf/docx/txt）
│       └── chunking.py              # 文本切块
├── test_chunking.py                 # 切块逻辑单测（不依赖网络/API Key）
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek（或其他 OpenAI 兼容接口）API Key：

```
LLM_API_KEY=你的实际key
```

> 用别的模型厂商同理，只需改 `LLM_BASE_URL` 和 `LLM_MODEL_NAME`，
> 比如通义千问的 base_url 是 `https://dashscope.aliyuncs.com/compatible-mode/v1`。

### 3. 启动服务

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. 测试

打开浏览器访问 **http://localhost:8000/docs**，会看到自动生成的交互式 API 文档，
在网页上就能直接测试，不需要额外写前端：

1. 展开 `POST /api/upload`，点 "Try it out"，上传一个 PDF/Word/TXT 文件
2. 展开 `POST /api/chat`，输入 `{"question": "文档里讲了什么？"}`，看返回的回答和引用来源

也可以用 curl：

```bash
curl -X POST "http://localhost:8000/api/upload" -F "file=@/path/to/your.pdf"

curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "文档主要讲了什么？"}'
```

## 验收自查清单（对应路线图第一阶段）

- [ ] 上传一份 PDF，能问出至少 5 个准确问题
- [ ] 回答里清楚标注引用来源（哪个文件）
- [ ] 问一个文档里没有的问题，模型会说"无法回答"而不是瞎编

## 第二阶段新增内容

- `POST /api/chat/stream`：流式问答接口（SSE），配合前端实现打字机效果
- `POST /api/sessions`、`GET /api/sessions`、`GET /api/sessions/{id}/messages`、`DELETE /api/sessions/{id}`：会话管理接口
- 新增 `app/services/session_service.py`：用 SQLite 持久化会话和消息记录
- 数据库文件会自动创建在 `data/app.db`（首次启动服务时）

## 第三阶段新增内容：知识图谱融合

- `app/services/extraction_service.py`：调用 LLM 从每个文本块抽取三元组（实体-关系-实体）
- `app/services/graph_service.py`：封装 Neo4j 的连接、写入、查询
- 文档上传时（`ingest_document`），除了存入向量库，还会逐块抽取三元组存入 Neo4j
- 问答时（`answer_question` / `stream_answer`），除了向量检索，还会判断问题里提到了哪些已知实体，查询它们在图谱里的关联关系，作为「知识图谱关联信息」一起提供给 LLM 参考——这就是「图谱增强检索」
- `GET /api/graph`：返回整个图谱的节点和关系数据，供前端可视化

### 需要额外启动的服务：Neo4j

需要先用 Docker 跑起来一个 Neo4j 实例：

```bash
docker run -d --name neo4j-agentic-rag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/你的密码 neo4j:5.24
```

启动后，在 `.env` 里配置对应的连接信息：

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的密码
```

可以打开 `http://localhost:7474` 用网页界面直接查看图谱内容（用同样的用户名密码登录），验证代码是否正确写入了数据。

### 关于抽取质量的说明

三元组抽取的准确率取决于 prompt 设计和文本本身的信息密度，不是每次都完美：
- 抽取失败或者格式不对时，代码会跳过这条数据，不会导致整个上传流程报错
- 同一个实体如果在文档里有不同叫法（比如"iPhone"和"苹果手机"），目前只在 prompt 层面要求模型尽量统一，没有做更复杂的实体对齐算法——这是可以在项目里继续优化、也是很好的"技术难点复盘"素材
- 建议先用信息密度高、结构清晰的文档（比如说明书、百科条目）测试，效果会比松散的叙述性文字更好

## 第四阶段新增内容：Agent 工具调用

- `app/services/agent_tools.py`：定义 3 个工具——`search_knowledge_base`（检索知识库）、`web_search`（联网搜索）、`generate_report`（生成 Markdown 报告文件）
- `app/services/agent_service.py`：Function Calling 循环——模型自主判断是否调用工具、调用哪个、调用几次，代码负责真正执行并把结果喂回模型，直到模型准备好给出最终回答
- `POST /api/agent/chat/stream`：Agent 对话的 SSE 接口，除了文字内容，还会推送 `tool_call`（开始调用某工具）、`tool_result`（工具执行结果）、`file`（生成了文件）等事件，前端据此展示"执行过程"
- 生成的文件存在 `data/workspace/{session_id}/` 下，通过 `/files/{session_id}/{filename}` 提供下载

### 关于联网搜索工具

`web_search` 工具依赖 [Serper.dev](https://serper.dev) 的搜索 API（免费额度每月 2500 次），需要单独注册申请 key，配置到 `.env` 的 `SERPER_API_KEY`。**不配置也没关系**——这个工具会返回"未配置"的提示信息，Agent 会据此调整回答策略（比如提示用户该功能不可用），不会导致程序报错崩溃。这是设计上刻意做的容错处理：外部依赖不可用时，系统应该优雅降级，而不是直接挂掉。

### 关于 MAX_TOOL_ITERATIONS

`agent_service.py` 里设了 `MAX_TOOL_ITERATIONS = 5`，防止模型陷入"一直觉得需要再调用一次工具"的死循环（既浪费 API 调用次数和费用，也会让用户等很久）。这是做 Agent 系统时一个容易被忽略但很重要的工程细节。

## 已知限制（留给第五阶段解决）

- 实体对齐（同一实体不同叫法归一化）还比较简单，依赖 prompt 而非专门算法
- 只支持文字版 PDF，扫描版（图片）PDF 解析不出文字，需要 OCR（后续可选）
- 没有用户登录/鉴权，单用户使用
- 对话上下文目前只是"存历史记录"，还没有把历史对话内容真正拼进 prompt 参与多轮理解（比如追问"那它呢"这种依赖上下文的问题还答不好）
- Agent 目前只有"检索/搜索/生成文件"三个工具，还没有代码执行沙盒、也没有多智能体编排（路线图里提到的进阶方向，可选）
- Agent 模式和知识库问答模式是分开的两条链路，还没有做成"根据问题自动判断该用哪种模式"的统一入口，这是可以在后续优化的点

## 常见问题

**Q: 第一次启动很慢？**
A: 正常，`sentence-transformers` 第一次运行会自动下载 embedding 模型（几十MB），下载完之后会缓存，之后启动就快了。

**Q: 上传后 chunk_count 是 0？**
A: 大概率是扫描版 PDF（图片形式），`pypdf` 提取不出文字。换一份文字版 PDF 测试。

**Q: LLM 调用报错 401？**
A: 检查 `.env` 里的 `LLM_API_KEY` 是否填对，以及余额是否充足。
