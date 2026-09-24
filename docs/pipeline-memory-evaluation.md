# Agentic RAG：SSE、会话记忆、工具容错与回答评估

本文对应 `feat/pipeline-sse` 分支。Router V5.2 原有规则、检索排序、Context Judge、Rewrite 采用条件未改。新功能使用相同管线结果驱动非流式与流式模式，避免两套决策分叉。

## 1. 三条问答路径

| 路径 | 检索和生成 | 会话上下文 | 过程显示 |
| --- | --- | --- | --- |
| Basic RAG | 一次 Chroma 检索；资料不足时提示用户 | 追问消歧、生成提示 | 检索中、来源、回答 |
| Agentic RAG | Router V5.2 → Planner → Actor → Reflector → 可选 Rewrite/Retry → 回答 | 追问消歧、生成提示 | 各节点实际开始或结束的事件 |
| Tool Agent | LLM Function Calling，可选知识库、Serper、报告文件 | 最近对话作为消息 | 每次工具调用/结果/错误、文件、回答 |

会话历史只帮助解释“它”“那个”“为什么”等跟进问题。生成提示明确标记历史不可作为事实来源；文档问题仍要通过 Chroma/Neo4j 检索核实。当前 SQLite 默认存最近 8 条消息，最多约 4000 字，单条最多 1000 字。消歧 LLM 最多返回 300 字，失败使用原问题；没有历史的首次提问不调用消歧 LLM。SQLite 使用外键、会话校验，并把一问一答同事务提交；网络断流或生成出错不会增加半轮历史。

## 2. 真正的 Pipeline SSE

服务端使用同步生成器并由 FastAPI `StreamingResponse` 迭代：每次 `yield` 时传出一帧，再开始下一步耗时操作。原实现调用整个 `run_retrieval_pipeline` 后才回放 Planner/Actor/Reflector；当前 `stream_retrieval_pipeline` 与非流式调用共享决策代码。Nginx `/api/` 关闭 `proxy_buffering`，后端还设置 `X-Accel-Buffering: no`。

| 事件 | 触发时点 | 主要字段 |
| --- | --- | --- |
| `stage(memory)` | 读取当前会话历史后、消歧前 | `stage`, `status` |
| `memory` | 消歧后 | `resolved_question`, `turns` |
| `stage(router)` | Router 实际运行前 | `stage`, `status` |
| `router` | Router 返回后 | `route`, `route_reason` |
| `planner` | 计划形成后 | 路线与步骤 |
| `stage(actor)` | 第 1 或第 2 次检索前 | `attempt` |
| `actor` | 对应检索完成后 | `attempt`, 命中数、图谱状态 |
| `stage(reflector)` / `reflector` | 证据判断前 / 判断后 | `round_index`, `sufficient`, `action` |
| `stage(rewrite)` / `rewrite` | 改写前 / 得到候选后 | `candidate` |
| `retrieval_info` | 最终检索上下文确定后 | 完整路线、改写与 trace 摘要 |
| `sources` | 生成前 | 文件、段号、检索分数 |
| `stage(answer)` / `content` | LLM 流式生成开始 / 每次返回 token | 文本片段 |
| `done` | 完整成功并保存会话后 | `status: ok` |
| `error` | 任何执行或保存异常 | 不含内部异常细节的提示 |

Basic RAG 跳过 Router/Planner/Reflector，只发 `stage(vector)`、`basic_info`、`sources`、`content` 和 `done`。DIRECT 不执行检索，但仍发 Router、Planner、Reflector 的结果。前端按 SSE 帧边界解析，支持跨网络分块与 CRLF；没有 `done` 的连接视为失败。`error` 不会被当成完成。操作步骤可以在 LLM 尚未开始回答时显示。模型若没有输出有效回答，服务端发送 `error`，不保存空轮次。

示例：

```text
data: {"type":"stage","data":{"stage":"actor","status":"running","attempt":1}}

data: {"type":"actor","data":{"attempt":1,"route":"vector","vector_hit_count":4,"graph_context_available":false}}

```

## 3. Tool Agent 容错

- JSON 参数必须符合白名单与长度要求；缺失、空字符串、额外字段、未知工具都返回工具错误，并作为 `tool` 消息交给模型解释。
- 一轮最多 5 次模型工具循环、8 次实际工具执行；相同工具与原始参数重复调用会停止执行并通知模型。
- OpenAI 兼容请求设 30 秒超时，客户端最多重试 2 次；Serper 请求设 10 秒超时，缺少 Key 会明确失败。工具异常不会将服务器堆栈发给浏览器。
- 工具结果送回模型的文本最多 3000 字，前端摘要最多 300 字；SSE `tool_call` 只带名称与调用 ID，避免把报告正文等大参数推给浏览器。
- 报告文件使用清洗过的标题与随机后缀，避免同名覆盖；生成文件只向前端返回文件名。
- 历史聊天消息传给 Tool Agent 只用于理解追问；知识库事实仍应调用检索工具核实。

这些措施限制了常见无限循环、无效参数和工具不可用场景。Tool Agent 不是具有沙盒的任意代码执行系统；当前只开放三个固定工具。

## 4. 回答层评估

`backend/evaluation/evaluate_answers.py` 从现有 `benchmark.csv` 取 `review_status=ready` 且有 `gold_answer` 的题。`live` 模式逐题执行当前 `answer_question`，记录真实来源、证据片段与单题耗时；`score` 模式只读已有预测 CSV，不调用 API，便于重复计算。按错误数、路由准确率、关键词召回、预期来源命中、知识题引用率、DIRECT 错误引用率、不可回答题拒答率、可回答题误拒答率和 P50/P95 延迟汇总。组合预期来源（文档 + 知识图谱）需要全部精确命中。报告包含样本数、成功数和失败数。

`--judge` 对每个成功回答提供参考答案和**本次实际检索证据**，由配置的 LLM 输出 `correct`、`faithful` 布尔值及理由。Judge 失败仅影响这题的 Judge 计数，仍保留其他规则指标。关键词召回是覆盖率诊断，不能代表回答正确；Judge 结果亦需人工抽检，特别是拒答和图谱事实。CSV 与 JSON 输出保存在 `backend/evaluation/results/answers/`，不纳入 Git。

```bash
cd backend
python -m evaluation.evaluate_answers --mode live --limit 20
python -m evaluation.evaluate_answers --mode live --limit 20 --judge
python -m evaluation.evaluate_answers --mode score --predictions evaluation/results/answers/answers.csv
```

评估开始前需上传与 Gold 来源一致的原始资料并确认 Chroma/Neo4j、LLM 配置；Gold 不能从模型输出中修补。Router 的冻结 Holdout-06 独立测试有单独协议，新脚本不改变该协议，也不能将 Holdout 再次调参后的结果称为独立成绩。

## 5. 验证状态与限制

2026-09-24，在无外部 API、无已导入知识库和 Neo4j 实例的隔离环境执行了离线后端测试与前端测试/构建。测试覆盖：SSE 在检索前发帧、流失败不写半轮、会话隔离、有界记忆与追问消歧、重复工具调用、参数校验、评估指标口径，以及跨网络分块 SSE 解析。真实数据回答质量、在线延迟与 Docker 全服务启动**尚未测量**；README 和简历不能写成线上生产性能。

现有接口面向本机单用户，无登录与权限隔离，静态文件下载路径同样未做用户鉴权；公网部署前需要补鉴权、权限、上传限额与多租户隔离。会话 trace 当前只在本次 SSE/前端内存展示，刷新后会保留问答和来源，但不会恢复完整执行过程。
