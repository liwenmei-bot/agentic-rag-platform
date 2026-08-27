"""
Agent 编排逻辑（Function Calling 循环）。

核心流程：
1. 把用户问题 + 工具列表 一起发给 LLM
2. LLM 要么直接回答，要么返回"我要调用某个工具，参数是这些"
3. 如果是后者：代码真正执行这个工具，把结果作为新的一条消息加入对话历史，
   再让 LLM 看着这个结果继续想（可能再调用别的工具，也可能这次准备直接回答了）
4. 重复第 2-3 步，直到 LLM 不再要求调用工具（或者达到最大循环次数，防止死循环）
5. 最后一步用流式方式生成最终的文字回答，实现打字机效果

这个循环本身就是"Agent"和普通"一问一答 RAG"最核心的区别——
普通 RAG 是固定流程（检索→回答），Agent 是模型自己决定"这一步该做什么"。
"""
import json
from pathlib import Path

from openai import OpenAI

from app.core.config import settings
from app.services.agent_tools import TOOL_SCHEMAS, execute_tool

_llm_client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

AGENT_SYSTEM_PROMPT = """你是一个能自主使用工具完成任务的智能助手。你可以：
1. 使用 search_knowledge_base 检索用户上传的文档知识库
2. 使用 web_search 联网搜索最新信息
3. 使用 generate_report 把内容整理成 Markdown 报告文件供用户下载

规则：
- 根据用户问题自主判断是否需要调用工具，不要为了调用而调用。简单的闲聊或常识问题可以直接回答。
- 如果需要多个步骤才能完成任务（比如先检索资料再生成报告），依次调用相应工具。
- 每次工具调用后，根据返回结果判断下一步该做什么。
- 最终回答时，清楚说明信息来源；如果生成了文件，明确告诉用户文件已生成。
- 工具调用失败或没有结果时，如实告知用户，不要编造。
"""

MAX_TOOL_ITERATIONS = 5


def run_agent(question: str, workspace_dir: Path):
    """
    生成器函数，逐步 yield 出 Agent 执行过程中的各类事件，供 SSE 推送给前端：
    - tool_call：模型决定调用某个工具
    - tool_result：工具执行完的结果摘要
    - file：生成了文件
    - content：最终回答的流式文字
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    reached_limit = True

    # 第一段循环：处理工具调用，直到模型不再要求调用工具，或者达到循环上限
    for _ in range(MAX_TOOL_ITERATIONS):
        response = _llm_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.3,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            reached_limit = False
            break

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            yield {"type": "tool_call", "data": {"name": tool_name, "arguments": arguments}}

            result_text, file_info = execute_tool(tool_name, arguments, workspace_dir)

            yield {"type": "tool_result", "data": {"name": tool_name, "result": result_text[:300]}}
            if file_info:
                yield {"type": "file", "data": file_info}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

    if reached_limit:
        yield {"type": "content", "data": "（已达到最大工具调用次数，基于目前已获得的信息回答）"}

    # 最后一步：不再传 tools 参数，强制模型只生成文字回答，并且用流式方式输出
    stream = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=messages,
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "content", "data": delta}
