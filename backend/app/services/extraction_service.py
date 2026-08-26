"""
三元组抽取：让 LLM 从一段文本里抽出 (主体, 关系, 客体) 结构。

Prompt 设计要点（路线图第三阶段提到的坑）：
1. 要求输出严格的 JSON，方便代码直接解析，不要让模型输出多余的解释文字
2. 给出具体的输出格式示例（few-shot），减少模型格式跑偏的概率
3. 限制每次抽取的数量，避免模型抽得太泛（什么都往里塞，图谱会变得很乱）
4. 明确要求实体名要"归一化"到最常见/最规范的说法，减少同一个实体因为
   叫法不同（"苹果公司" vs "Apple"）而被存成两个节点的问题
   （完全解决这个问题需要更复杂的实体对齐算法，这里先用 prompt 层面缓解）
"""
import json

from openai import OpenAI

from app.core.config import settings

_llm_client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

EXTRACTION_PROMPT = """你是一个信息抽取助手，任务是从下面的文本中抽取实体关系三元组。

规则：
1. 只抽取文本中明确表达的事实关系，不要推测或编造。
2. 每个三元组的格式是：主体、关系、客体，都必须是简洁的名词或短语（关系一般是2-4个字的动词短语，比如"创始人"、"位于"、"属于"）。
3. 同一个实体在整段文本里如果有不同叫法，统一使用最规范、最常见的那个名称。
4. 最多抽取 8 条最重要的三元组，不要为了凑数抽取无意义的关系。
5. 严格按照下面的 JSON 格式输出，不要输出任何其他文字、解释或 Markdown 代码块标记。

输出格式示例：
[{"subject": "苹果公司", "relation": "创始人", "object": "史蒂夫·乔布斯"}, {"subject": "史蒂夫·乔布斯", "relation": "出生于", "object": "美国"}]

文本内容：
{text}

请输出 JSON：
"""


def extract_triples(text: str) -> list[dict]:
    """从一段文本抽取三元组列表，抽取失败或解析失败时返回空列表（不中断主流程）。"""
    if not text.strip():
        return []

    prompt = EXTRACTION_PROMPT.replace("{text}", text)

    try:
        response = _llm_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 抽取任务要稳定、可重复，温度要低
        )
        raw_output = response.choices[0].message.content.strip()

        # 兜底处理：万一模型还是输出了 ```json 包裹的代码块，去掉包裹符号再解析
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]

        triples = json.loads(raw_output)
        if not isinstance(triples, list):
            return []
        return triples

    except (json.JSONDecodeError, Exception) as e:
        # 抽取失败不应该导致整个文档上传流程失败，打印日志、返回空列表即可
        print(f"三元组抽取失败: {e}")
        return []
