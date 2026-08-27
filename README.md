# agentic-rag-platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![Vue](https://img.shields.io/badge/Vue-3.5-42b883.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.24-018bff.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-All%20Phases%20Complete-brightgreen.svg)

一个融合 **RAG 检索、知识图谱、Agent 工具调用** 的智能知识平台，支持 Docker 一键部署。
参考 [xerrors/Yuxi](https://github.com/xerrors/Yuxi) 等开源项目的架构思路，独立设计并实现。

## 这个项目能做什么

1. **上传文档**（PDF / Word / TXT），系统自动解析、切块、向量化，构建可检索的知识库
2. **提问**，系统检索相关内容并生成带引用来源的回答，支持流式打字机输出
3. 上传文档时，系统还会自动**抽取实体关系**，构建可视化的知识图谱；问答时结合图谱关系做更深层的关联推理
4. 切换到 **Agent 模式**，模型能自主判断是否需要检索知识库、联网搜索、或生成一份可下载的报告文件，并自主执行

## 功能演示

<!-- 把测试时的截图放进 docs/screenshots/ 目录，然后取消下面的注释替换成真实图片 -->
<!--
| 知识库问答 | 知识图谱可视化 |
|---|---|
| ![chat](./docs/screenshots/chat.png) | ![graph](./docs/screenshots/graph.png) |

| Agent 工具调用 | Docker 一键部署 |
|---|---|
| ![agent](./docs/screenshots/agent.png) | ![docker](./docs/screenshots/docker.png) |
-->

## 技术栈

| 层 | 技术选型 |
|---|---|
| 前端 | Vue 3 · Vite · Pinia · ECharts |
| 后端 | FastAPI · SSE 流式响应 |
| 向量检索 | Chroma · sentence-transformers (bge-small-zh) |
| 知识图谱 | Neo4j · Cypher |
| 会话存储 | SQLite |
| LLM | DeepSeek API（OpenAI 兼容接口，可替换为其他厂商） |
| 部署 | Docker · Docker Compose · Nginx |

## 开发进度

- [x] 第一阶段：基础 RAG 问答（文档解析 → 切块 → 向量检索 → 带引用回答）
- [x] 第二阶段：Vue 3 对话式前端 + 流式输出 + 多会话管理
- [x] 第三阶段：知识图谱融合（Neo4j + 图谱增强检索 + 可视化）
- [x] 第四阶段：Agent 工具调用（知识库检索、联网搜索、生成报告文件）
- [x] 第五阶段：Docker Compose 一键部署

## 项目结构

```
agentic-rag-platform/
├── backend/              # FastAPI 后端：RAG、知识图谱、Agent 全部逻辑
│   ├── app/
│   │   ├── api/           # 接口路由
│   │   ├── services/      # 核心业务逻辑
│   │   └── utils/         # 文档解析、切块工具
│   └── Dockerfile
├── frontend/             # Vue 3 前端
│   ├── src/
│   │   ├── components/    # 对话窗口、侧边栏、图谱可视化
│   │   └── stores/        # Pinia 状态管理
│   ├── Dockerfile
│   └── nginx.conf         # 生产环境的反向代理配置
├── docker-compose.yml     # 一键编排 frontend + backend + neo4j
└── LICENSE
```

## 快速开始：Docker 一键部署（推荐）

**前提**：安装好 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。

```bash
git clone https://github.com/liwenmei-bot/agentic-rag-platform.git
cd agentic-rag-platform

cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、NEO4J_PASSWORD（SERPER_API_KEY 可选）

docker compose up --build
```

首次构建会花几分钟（需要下载 Python/Node/Neo4j 镜像，以及 sentence-transformers 依赖的 torch），构建完成后：

- 前端页面：**http://localhost**
- 后端 API 文档：http://localhost:8000/docs
- Neo4j 管理界面：http://localhost:7474

所有数据（向量库、会话记录、图谱数据、Agent 生成的文件）都通过 Docker volume 持久化，容器重启不会丢失。

## 快速开始：本地开发模式

如果你想修改代码、逐步调试，而不是直接用 Docker 部署，参考各子项目的详细文档：

- 后端：[`backend/README.md`](./backend/README.md)
- 前端：[`frontend/README.md`](./frontend/README.md)

本地开发模式下，Neo4j 需要单独用 Docker 起一个容器（不经过 docker-compose），具体步骤同样在 `backend/README.md` 里。

## 开发历程

这个项目从零开始按照"最小可用 → 逐步加功能"的思路分五个阶段完成，每个阶段都有独立的技术验证。感兴趣可以看各阶段目录下的代码演进（通过 git log 查看提交历史），或者查看每个子项目 README 里"已知限制"部分——这些是真实踩过的坑和可以继续优化的方向。

## License

本项目基于 [MIT License](./LICENSE) 开源。
