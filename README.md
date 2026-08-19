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

## 项目结构

```
agentic-rag-platform/
├── backend/          # 后端服务（FastAPI + RAG 核心逻辑），详见 backend/README.md
└── LICENSE
```

## 快速开始

详细的安装、配置、启动步骤请见 [`backend/README.md`](./backend/README.md)。

## License

本项目基于 [MIT License](./LICENSE) 开源。
