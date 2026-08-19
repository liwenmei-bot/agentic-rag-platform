# agentic-rag-platform

一个融合 RAG 检索、知识图谱与 Agent 工具调用的智能知识平台（开发中）。
参考 [xerrors/Yuxi](https://github.com/xerrors/Yuxi) 等开源项目的架构思路，独立设计并实现。

技术栈：FastAPI · Chroma · sentence-transformers · DeepSeek API（后续阶段将加入 Vue 3 前端、Neo4j 知识图谱、Agent 工具调用）

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

## 已知限制（这些是第二、三阶段要解决的，第一阶段先不管）

- 只支持非流式回答（一次性返回），流式打字机效果留到做前端时再加
- 没有多会话/上下文记忆，每次提问都是独立的
- 只能全库检索，还没做基于知识图谱的增强检索
- 只支持文字版 PDF，扫描版（图片）PDF 解析不出文字，需要 OCR（后续可选）

## 常见问题

**Q: 第一次启动很慢？**
A: 正常，`sentence-transformers` 第一次运行会自动下载 embedding 模型（几十MB），下载完之后会缓存，之后启动就快了。

**Q: 上传后 chunk_count 是 0？**
A: 大概率是扫描版 PDF（图片形式），`pypdf` 提取不出文字。换一份文字版 PDF 测试。

**Q: LLM 调用报错 401？**
A: 检查 `.env` 里的 `LLM_API_KEY` 是否填对，以及余额是否充足。
