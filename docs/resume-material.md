# 简历素材：Agentic RAG 知识库问答与评估平台

以下文字以当前代码和离线测试为依据，适合 AI 应用开发/大模型应用开发岗位。使用前可根据真实上线与评估结果继续更新；不要把离线回归、历史 Router 成绩或未运行的回答评估写成线上效果。

## 简历版（约 180 字）

**Agentic RAG 知识库问答与评估平台｜独立开发**
技术栈：Python、FastAPI、Vue 3、Chroma、Neo4j、SQLite、OpenAI 兼容接口、SSE、Docker Compose。

- 实现 PDF/DOCX/TXT 文档解析、切块、向量入库与图谱关系抽取，提供基础 RAG 和 DIRECT/VECTOR/GRAPH/HYBRID 四路路由的 Agentic RAG 问答，回答支持来源定位。
- 将检索管线改为逐阶段 SSE：Router、Planner、Actor、Reflector、Query Rewrite 在实际执行时发事件，前端显示检索进度与流式答案；按会话保存完整问答，用有界历史解析多轮追问。
- 加固 Tool Agent 的 Function Calling：参数校验、超时与重试、重复调用拦截、执行次数限制、错误反馈及报告文件下载；增加回答层评估脚本和离线回归测试。

## 面试时如何说清边界

| 问题 | 依据当前代码的回答 |
| --- | --- |
| “SSE 真流式在哪里？” | 同步生成器在 Router/Actor/Reflector 等耗时步骤之前和之后分别 `yield`，Nginx 关闭代理缓冲；回答 token 也单独发送。 |
| “多轮记忆如何实现？” | SQLite 保存完整成功轮次；读取最近 8 条、最多 4000 字，短追问用 LLM 消歧。历史只帮助理解指代，知识库事实仍由检索证据支撑。 |
| “Tool Agent 怎么防失控？” | 工具参数白名单、每轮调用上限、重复签名去重、超时/重试、工具错误回传模型。它是固定工具的 Function Calling，不支持任意代码执行。 |
| “回答准确率多少？” | 当前没有在这个功能分支完成真实回答评测，不能报数字。可展示现有评估集、脚本和输出指标；评估后区分关键词诊断、LLM Judge 与人工抽检。 |
| “Router V5.2 呢？” | 旧版规则及冻结 Holdout 协议保留。新 SSE/Memory 工作没有产生新的 Router 独立 Holdout 成绩；不能把开发回归结果称作独立泛化成绩。 |
| “上线了吗？” | 完成本地代码、Docker Compose 配置和离线自动化测试，尚未验证当前环境完整外部服务联调或公网生产部署。 |

## 可演示路径

1. 上传包含实体关系的文档，先用 Basic RAG 提问，看来源。
2. 换 Agentic RAG 提问，观察 Route → Planner → Actor → Reflector 的逐阶段状态及答案；问“它呢”展示同会话消歧。
3. 用 Tool Agent 生成报告，观察工具调用与下载；模拟重复调用/缺少 API Key 看错误状态。
4. 运行 `python -m pytest -q tests/`、`npm test`，解释 SSE 事件顺序和完整轮次写入测试。
5. 有真实数据、API Key 和 Neo4j 时，运行回答评估脚本，并明确数据集、样本数与指标口径。
