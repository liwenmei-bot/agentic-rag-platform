"""
RAG 核心逻辑：
1. 文档入库：解析 -> 切块 -> 存入向量库 -> 抽取三元组存入知识图谱
2. 问答：Agent Router 检索管线 -> 拼接 prompt -> 调用 LLM -> 返回带来源的回答

Agent Router 检索管线（这一版新加的核心能力）：

    问题
      │
      ▼
  是否是闲聊/无需检索？──是──▶ 直接回答
      │否
      ▼
  向量检索 + 图谱检索
      │
      ▼
  上下文是否充分？──是──▶ 生成回答（带引用）
      │否
      ▼
  基于首次检索证据做保守查询改写（不允许凭空补概念）→ 用改写后的查询再检索一次
      │
      ▼
  生成回答（带引用）

这套"检索->判断->必要时改写重试"的模式，是 Agentic RAG 和普通 RAG 的核心区别——
普通 RAG 检索一次就直接生成，检索效果差的时候模型只能"将就着回答"；
这里加了一层自我判断和纠正的能力，遇到检索效果不好的情况，会先尝试换个问法再查一次，
而不是原样把不充分的资料交给生成模型。详细设计说明见 docs/architecture.md。
"""
import json
import re
import uuid

from openai import OpenAI

from app.core.config import settings
from app.services import graph_service
from app.services.extraction_service import extract_triples
from app.services.vector_store_service import add_chunks, search
from app.services.agent_orchestrator import (
    build_actor_trace,
    build_plan,
    build_reflection,
)
from app.utils.chunking import chunk_text
from app.utils.parsing import parse_document

_llm_client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

# Prompt 设计要点（对应路线图里提到的"防止模型瞎编"）：
# 1. 明确要求"只根据资料回答"
# 2. 明确要求"找不到就说找不到"，不许编造
# 3. 要求标注引用来源，方便用户核实（这一版升级成"文件名 + 第N段"，定位更精确）
# 4. 第三阶段新增：允许模型参考"知识图谱关联信息"做更深层的关联回答
SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请严格遵守以下规则：

1. 只能根据下面提供的【参考资料】和【知识图谱关联信息】回答问题。
   不要使用自己的外部知识补充资料中没有提供的事实、关系、原因或机制。

2. 只有当【参考资料】和【知识图谱关联信息】都确实不足以回答用户问题时，
   才回答“根据现有资料无法回答这个问题”。
   不要因为缺少实体的背景定义，就否定知识图谱已经明确给出的关系。

3. 如果【知识图谱关联信息】直接给出了：
       A -[关系R]-> B
   而用户询问：
       “A 和 B 是什么关系？”
       “A 与 B 如何关联？”
       “A 对 B 是什么关系？”
   那么这条图谱边本身就是充分证据。

   此时应直接按照关系方向自然表述，例如：
       Agent -[执行方式]-> 多轮执行
   应回答：
       “多轮执行是 Agent 的一种执行方式。【来源：知识图谱】”

   不需要额外要求知识库提供 A、B 或“关系R”的定义。

4. 对知识图谱关系进行自然语言转换时，必须保持方向和含义：
       A -[包含]-> B
       → “A 包含 B”
       A -[执行方式]-> B
       → “B 是 A 的执行方式”
       A -[约束对象]-> B
       → “A 的约束对象是 B”
   不要把关系方向反过来，也不要把一种关系改写成另一种因果或定义关系。

5. 如果用户问的是“为什么”“原因是什么”“具体机制是什么”“会产生什么影响”等解释性问题，
   而知识图谱只提供一条关联边、没有提供足够的原因或机制证据：
   - 可以先回答图谱明确支持的关系；
   - 然后明确说明更深层的原因/机制在现有资料中没有提供；
   - 禁止根据常识自行补全因果链。

6. 如果参考资料与知识图谱都提供了证据，可以综合回答。
   文档负责补充具体说明，知识图谱负责补充实体关系；
   如果二者冲突，不要自行裁决，应该指出现有资料存在不一致。

7. 文档内容的引用格式：
       【来源：文件名 第N段】

8. 知识图谱关系的引用格式：
       【来源：知识图谱】

9. 回答要直接、简洁、准确。
   优先先回答用户真正问的结论，再补充必要说明。
   不要机械重复参考资料，也不要为了显得谨慎而对已经被资料直接支持的结论拒答。
"""

DIRECT_SYSTEM_PROMPT = """你是一个简洁、友好的通用助手。
当前问题被 Agent Router 判断为无需知识库检索的 DIRECT 请求。
请直接回答用户，不要伪造知识库引用，也不要声称查过文档或知识图谱。
"""

BASIC_RAG_SYSTEM_PROMPT = """你是一个严谨的基础知识库问答助手。
当前模式是 Basic RAG，只允许使用提供的【参考资料】回答。

规则：
1. 不使用知识图谱，不调用 Agent Router，不做 Query Rewrite。
2. 不要使用外部知识补充资料中没有提供的事实。
3. 如果参考资料不足，明确回答“根据现有知识库资料无法回答这个问题”。
4. 文档引用格式：
   【来源：文件名 第N段】
5. 回答直接、简洁、准确。
"""

ROUTER_PROMPT = """你是 Agentic RAG 的四路路由器。你的任务是判断：
为了回答当前问题，系统应该使用哪一种信息来源/处理方式。

可选路线：
- direct：不需要知识库证据的“任务型请求”，例如寒暄、感谢、告别、简单翻译、简单算术、轻量创作、简单文本转换。
- vector：主要需要文档文本证据的“知识型问题”，例如定义、事实、数值、属性、参数、原因、机制、影响、项目配置、部署信息、成本、延迟、用户数、型号等。
- graph：主要需要实体关系证据，例如关系、联系、连接、指向、上下游、归属、关系标签、哪个节点、谁导致谁、谁决定谁。
- hybrid：明确需要“文本证据 + 图谱关系”两类证据共同完成回答。

重要原则：
1. DIRECT 不是“通用知识问答”路线。即使你凭常识知道答案，只要用户是在询问技术/项目/知识事实，仍然应该按证据类型选择 VECTOR / GRAPH / HYBRID。
2. “系统提示词会占用上下文窗口吗？”、“工具调用结果会不会让上下文变长？”、“长上下文会影响成本和延迟吗？”、“项目部署在哪家云服务商？”都属于知识/事实/机制/属性查询，通常走 VECTOR。
3. 如果问题明确询问 A 与 B 的关系、联系、连接、边、上下游、归属、对应、关系类型，应优先 GRAPH。
4. HYBRID 必须有明确的双证据需求。下列表达都表示双证据：“先查 README 再沿 Neo4j”“代码说明 + Neo4j 边”“文本事实也需要图谱边”“文档内容与关系路径两个角度”“别只看图谱也别只看文档，两边都用”。 仅仅因为问题涉及多个技术概念，或者一个关系也可以用文字解释，不足以选择 HYBRID。
5. 如果一句话先寒暄，后面又提出知识问题，应忽略寒暄前缀，按照知识问题路由。
6. 显式证据约束最高优先：
   - 只看文档 / 不看图谱 -> vector
   - 只看图谱 / 不查文档 -> graph
   - 两边证据都要 -> hybrid
7. 无法确定时：普通知识事实/解释 -> vector；明确实体关系 -> graph；明确双证据 -> hybrid；只有真正不需要知识证据的任务才 direct。

只输出 JSON，不要 Markdown，不要额外解释：
{{"route":"vector","reason":"用户询问知识事实，需要文档文本证据"}}

用户问题：
{question}
"""

REWRITE_PROMPT = """你是一个“保守型”知识库检索查询优化助手。

你的任务不是扩写问题，而是在【不改变用户原意】的前提下，让查询更适合检索。

必须严格遵守：
1. 只能使用【原问题】以及【首次检索候选资料】里已经出现或可以直接确定的实体、概念和关系。
2. 对“这个、这种东西、它、这样”等模糊指代，只能根据候选资料消歧；如果候选资料不能确定，就保留模糊表达，不要猜。
3. 禁止凭空补充候选资料中没有出现的新技术概念、新机制、新例子或新因果关系。
4. 禁止为了“显得专业”而加入额外术语。例如资料没有出现“类型系统、熵增、状态发散”等词，就绝不能自行加入。
5. 保留原问题的问题类型和范围：
   - 原问题问“为什么”，改写后仍然只问原因；
   - 原问题问“是什么”，不要扩展成优缺点、机制、应用等；
   - 不要擅自增加“请说明机制、影响因素、典型表现”等额外要求。
6. 优先做“最小改写”：替换口语表达、补上资料中明确出现的核心实体或关键词即可。
7. 输出必须是一句话，只输出改写后的检索查询，不要解释，不要加“改写后：”之类的前缀。

【原问题】
{question}

【首次检索候选资料】
{evidence}

【改写后的检索查询】
"""

# =============================================================================
# Router V4: explicit constraints + high-confidence direct + intent scoring
# =============================================================================

def _matches_any(patterns: tuple[str, ...], question: str) -> bool:
    return any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in patterns)


# ----- DIRECT: only very high-confidence non-KB tasks -----

_SOCIAL_ONLY_PATTERNS = (
    r"^(?:你好|您好|你好呀|嗨|哈喽|hello|hi|早安|早呀|早上好|上午好|中午好|下午好|晚上好|晚安)[!！。.，,？?\s]*$",
    r"^(?:(?:你好|早安|早上好|上午好|中午好|下午好|晚上好)[，, ]*)?(?:今天心情不错|最近怎么样|最近好吗|你好吗)[!！。.，,？?\s]*$",
    r"^(?:你在吗|在吗|在不在呀|在不在)[!！。.，,？?\s]*$",
    r"^(?:谢谢|多谢|谢啦|感谢|辛苦了|辛苦啦|好的[，, ]*谢谢|收到)(?:[，, ]*(?:今天)?(?:先到这里|先这样|回头见|晚点见|晚点聊))?[!！。.，,\s]*$",
    r"^(?:先这样|再见|拜拜|bye|回见|回头见|晚点聊|晚点见|我先走了[，, ]*回头见)[!！。.，,\s]*$",
    r"^(?:我(?:先)?去忙了|我先走了)(?:[，, ]*(?:之后|回头|晚点)(?:再)?聊|[，, ]*回头见)?[!！。.，,\s]*$",
    r"^(?:哈喽|你好|嗨)[，, ]*今天就不(?:问|聊)技术问题了[!！。.，,\s]*$",
)

_SIMPLE_TRANSLATION_PATTERNS = (
    r"^(?:请)?(?:帮我)?把.{1,120}(?:翻译成|翻成)(?:英文|英语|中文|汉语|日文|日语|韩文|韩语)[。.!！]?$",
    r"^(?:请)?翻译[：:].{1,160}$",
    r"^.{1,120}(?:中文|英文|英语|日文|日语|韩文|韩语)怎么说[？?。.！!]*$",
)

_SIMPLE_ARITHMETIC_PATTERN = re.compile(
    r"^\s*[-+]?\d+(?:\.\d+)?"
    r"(?:\s*(?:再\s*)?(?:\+|\-|\*|/|×|÷|加|减|乘以|乘|除以|除)\s*[-+]?\d+(?:\.\d+)?)+"
    r"\s*(?:等于多少|是多少|等于几|是几|结果是多少|结果是几)?[？?。.！!]*\s*$"
)

_LIGHT_CREATIVE_PATTERNS = (
    r"^(?:请)?(?:帮我)?(?:写|想|给我想)(?:一句|一条|一个|个)?(?:不超过.{0,10})?(?:简短的|礼貌的)?"
    r".{0,30}(?:祝福|口号|结束语|欢迎语|开场白|问候|鼓励语?)[。.!！]?$",
    r"^(?:请)?(?:帮我)?写一句.{0,50}(?:祝福|问候|鼓励语?|开场白|结束语|欢迎语)[。.!！]?$",
)

_SIMPLE_TEXT_TASK_PATTERNS = (
    r"^把.{1,100}(?:改成|转换成)(?:小写|大写)[。.!！]?$",
    r"^把.{1,120}(?:中的)?字母改成(?:小写|大写)[。.!！]?$",
    r"^把.{1,120}(?:的)?(?:两个)?单词首字母大写[。.!！]?$",
    r"^把.{1,120}按(?:逗号|空格|分号|顿号)(?:拆|分)成.{1,40}[。.!！]?$",
)


def _is_chitchat(question: str) -> bool:
    return _matches_any(_SOCIAL_ONLY_PATTERNS, question.strip())


def _is_direct_non_kb_task(question: str) -> bool:
    q = question.strip()
    return (
        _is_chitchat(q)
        or bool(_SIMPLE_ARITHMETIC_PATTERN.match(q))
        or _matches_any(_SIMPLE_TRANSLATION_PATTERNS, q)
        or _matches_any(_LIGHT_CREATIVE_PATTERNS, q)
        or _matches_any(_SIMPLE_TEXT_TASK_PATTERNS, q)
    )


# ----- V5 structured evidence-source parser -----

_TEXT_EVIDENCE_STRONG_PATTERNS = (
    r"\breadme\b",
    r"配置(?:文件|文档|说明)",
    r"代码说明",
    r"项目说明",
    r"说明文档",
    r"概念说明",
    r"文档(?:里|中|内容|片段|说明)",
    r"资料(?:里|中|中的|片段)?",
    r"正文",
    r"段落",
    r"文字(?:材料|证据|说明|定义|来源)",
    r"文本(?:事实|证据|说明|内容)",
    r"知识库(?:片段|内容)?",
    r"检索(?:到的)?(?:片段|段落)",
)

_GRAPH_EVIDENCE_PATTERNS = (
    r"知识图谱",
    r"图谱",
    r"知识图",
    r"neo4j",
    r"关系图",
    r"图结构",
    r"关系网络",
    r"图谱边",
    r"关系边",
    r"实体关系",
    r"实体连接",
    r"关系路径",
    r"组件关系(?:网络)?",
    r"服务依赖",
    r"图(?:里|中|里的|中的).{0,30}(?:关系|连接|节点|路径|边|上游|下游)",
)

_DUAL_EVIDENCE_CUES = (
    "同时", "结合", "综合", "一起", "一并", "两边", "两个角度",
    "合起来", "合并", "联合", "放在一起", "放在同一个答案",
    "一边", "也要", "也需要", "再用", "再从", "再沿", "再去",
    "先从", "先查", "先用", "最后合并", "都用", "共同", "分别", "负责",
)

_DOC_ONLY_PATTERNS = (
    r"(?<!不)(?<!别)(?<!要)(?:只|仅)(?:根据|看|查|参考|使用|用|从)?"
    r"(?:文档|资料|知识库|正文|文字证据|文本证据|文档片段|知识库片段|readme|项目说明|配置文件|说明文档)",
)

_GRAPH_ONLY_PATTERNS = (
    r"(?<!不)(?<!别)(?<!要)(?:只|仅)(?:根据|看|查|参考|使用|用|沿)?"
    r"(?:知识图谱|图谱|neo4j|关系图|知识图|图谱边|关系边)",
)

_DOC_EXCLUDE_PATTERNS = (
    r"(?:不|不要|别)(?:只)?(?:再)?(?:使用|用|查|看|参考|翻|依据)?"
    r"(?:文档|资料|知识库|正文|readme|项目说明|说明文档)",
)

_GRAPH_EXCLUDE_PATTERNS = (
    r"(?:不|不要|别)(?:只)?(?:再)?(?:使用|用|查|看|参考|沿)?"
    r"(?:知识图谱|图谱|neo4j|关系图|知识图)",
)

_DOC_FACT_REQUEST_PATTERNS = (
    r"(?:配置文件|配置文档|readme|文档|资料|项目说明|说明文档).{0,35}"
    r"(?:有没有|是否|写|记录|说明|给出|列出|使用|采用|配置|地址|端口|框架|工具|数值)",
    r"(?:有没有|是否).{0,35}(?:记录|写|说明|给出).{0,35}"
    r"(?:地址|端口|url|uri|型号|配置|框架|构建工具|服务商|账单|峰值)",
)


def _has_text_evidence_reference(question: str) -> bool:
    q = question.strip().lower()
    return _matches_any(_TEXT_EVIDENCE_STRONG_PATTERNS, q)


def _has_text_reference_broad(question: str) -> bool:
    q = question.strip().lower()
    return (
        _has_text_evidence_reference(q)
        or any(term in q for term in ("文档", "资料", "文字", "文本", "readme", "知识库", "说明"))
    )


def _has_graph_evidence_reference(question: str) -> bool:
    q = question.strip().lower()
    return _matches_any(_GRAPH_EVIDENCE_PATTERNS, q)


def _has_dual_evidence_intent(question: str) -> bool:
    q = question.strip().lower()

    has_text = _has_text_reference_broad(q)
    has_graph = _has_graph_evidence_reference(q)

    if not (has_text and has_graph):
        return False

    if any(cue in q for cue in _DUAL_EVIDENCE_CUES):
        return True

    # Sequential two-source instructions.
    sequential_patterns = (
        r"(?:先|先从|先查|先用).{0,80}(?:文档|资料|readme|说明|正文|文字|文本|知识库)"
        r".{0,100}(?:再|然后|最后|再沿|再去|再从|再用).{0,80}"
        r"(?:知识图谱|图谱|知识图|neo4j|关系图|图结构|关系网络|关系边|关系路径)",
        r"(?:一边).{0,80}(?:文档|资料|readme|说明|正文|文字|文本)"
        r".{0,100}(?:一边).{0,80}(?:图谱|neo4j|关系图|关系网络|关系边)",
        r"(?:需要).{0,60}(?:文本|文字|文档|资料).{0,80}(?:也需要|还需要).{0,60}"
        r"(?:图谱|neo4j|关系边|图谱边)",
    )
    return _matches_any(sequential_patterns, q)


def _explicit_modality_route(question: str) -> tuple[str, str] | None:
    q = question.strip()
    q_lower = q.lower()

    # Dual evidence always wins, including "别只看 A，也别只看 B，两边都用".
    if _has_dual_evidence_intent(q_lower):
        return "hybrid", "V5证据源解析：用户明确要求同时使用文本证据与图谱关系。"

    doc_excluded = _matches_any(_DOC_EXCLUDE_PATTERNS, q_lower)
    graph_excluded = _matches_any(_GRAPH_EXCLUDE_PATTERNS, q_lower)

    if graph_excluded and not doc_excluded:
        return "vector", "V5证据源解析：用户排除图谱证据，使用文档检索。"

    if doc_excluded and not graph_excluded:
        return "graph", "V5证据源解析：用户排除文档证据，使用知识图谱。"

    doc_only = _matches_any(_DOC_ONLY_PATTERNS, q_lower)
    graph_only = _matches_any(_GRAPH_ONLY_PATTERNS, q_lower)

    if doc_only and not graph_only:
        return "vector", "V5证据源解析：用户明确只使用文档/资料证据。"

    if graph_only and not doc_only:
        return "graph", "V5证据源解析：用户明确只使用图谱/关系证据。"

    return None


# ----- V5 intent scoring -----

_GRAPH_FEATURES: tuple[tuple[int, str], ...] = (
    (5, r"(?:知识图谱|图谱|关系图|知识图).{0,25}(?:边|节点|关系|联系|连接|指向|对应|上游|下游|归属|包含|导致|决定|路径)"),
    # Neo4j is a product name too, so "Neo4j 连接地址" must NOT be treated as an entity relationship.
    (5, r"neo4j.{0,25}(?:边|节点|关系|联系|连接(?!地址|字符串|配置|端口|url|uri)|指向|对应|上游|下游|归属|包含|导致|决定|路径)"),
    (5, r"(?:哪个|哪一个|什么)(?:节点|实体).{0,30}(?:连接|指向|对应|连|关系|上游|下游|流向)"),
    (5, r"(?:上游|下游)(?:实体|节点|步骤)|后继节点|关系标签|关系类型|哪一种连接|哪种连接"),
    (5, r"(?:前后怎么衔接|怎样衔接|如何衔接|怎么衔接|依赖关系|怎样的依赖|存在怎样的依赖)"),
    (5, r"(?:流向哪个|流向哪里|经过哪些节点|触发.{0,30}|谁接收|由谁接收|交给.{0,30}(?:模块|组件)|如何汇合|怎么汇合)"),
    (5, r"(?:位于谁之前|谁之前.{0,20}谁之后|位于.{0,20}之前.{0,20}之后|前后接哪些节点|关系路径)"),
    (4, r"什么关系|有何关系|关系是什么|之间.*关系|什么联系|有何联系|是什么联系|之间.*联系|有什么联系"),
    (4, r"如何关联|怎么关联|怎么连起来|如何连起来|怎么连接|如何连接|怎么挂接|如何挂接"),
    (4, r"指向了|指向谁|谁指向|连过来|连出去|挂接|归到|上位概念|下位概念"),
    (4, r"通过.{0,25}(?:关系|边|执行方式).{0,25}(?:连接|关联|连|指向)"),
    (4, r"谁.{0,20}(?:导致|决定)|由.{0,30}(?:导致|决定)"),
    (4, r"是不是一回事|是否是一回事|是不是同一个|是否相同|是否一样|同义概念"),
    (3, r"定义关系|包含关系|衡量关系|执行方式|关系网络|实体关系|实体连接"),
    (3, r"(?:对应|连接|关联).{0,30}(?:哪个|什么)(?:节点|实体|概念|属性|关系)"),
)

_VECTOR_FEATURES: tuple[tuple[int, str], ...] = (
    (6, r"(?:文档|资料|知识库|正文|段落|readme|配置文件|配置文档|项目说明|代码说明|说明文档)"
        r".{0,35}(?:有没有|是否|写|记录|提到|说明|描述|给出|查到|列出|配置|使用|采用)"),
    (4, r"什么是|是什么概念|什么意思|定义|具体表示什么|表示的是什么|具体指"),
    (4, r"为什么|原因|怎么解释|如何解释|有什么影响|会有什么影响"),
    (4, r"多少|多大|多长|多久|价格|费用|人数|用户数|日活|型号|显卡配置|概率|准确率|延迟|营业收入|营收"),
    (4, r"连接地址|连接字符串|url|uri|端口|web框架|构建工具|服务商|服务器账单|实测峰值"),
    (3, r"文档|资料|知识库|正文|段落|文字说明|文本说明|检索片段|readme|项目说明|配置文件"),
    (2, r"有没有|是否说明|是否记录|怎么描述|如何描述|哪几类|哪些内容|什么量级"),
)


def _intent_scores(question: str) -> tuple[int, int]:
    q = question.strip().lower()

    graph_score = sum(
        weight for weight, pattern in _GRAPH_FEATURES
        if re.search(pattern, q, flags=re.IGNORECASE)
    )
    vector_score = sum(
        weight for weight, pattern in _VECTOR_FEATURES
        if re.search(pattern, q, flags=re.IGNORECASE)
    )

    return graph_score, vector_score


def _high_confidence_intent_route(question: str) -> tuple[str, str] | None:
    q = question.strip().lower()

    # Structured document fact/config request overrides product-name relation noise.
    if (
        _has_text_evidence_reference(q)
        and _matches_any(_DOC_FACT_REQUEST_PATTERNS, q)
        and not _has_explicit_relation_intent(q)
    ):
        return "vector", "V5文档事实保护：问题明确询问配置/属性/记录信息。"

    graph_score, vector_score = _intent_scores(q)

    if graph_score >= 5 and graph_score - vector_score >= 2:
        return "graph", f"V5意图评分：GRAPH={graph_score}, VECTOR={vector_score}，关系意图高置信度。"

    if vector_score >= 5 and vector_score - graph_score >= 2:
        return "vector", f"V5意图评分：GRAPH={graph_score}, VECTOR={vector_score}，文档事实意图高置信度。"

    if graph_score >= 4 and vector_score == 0:
        return "graph", f"V5意图评分：GRAPH={graph_score}, VECTOR=0，关系意图明确。"

    if vector_score >= 4 and graph_score == 0:
        return "vector", f"V5意图评分：GRAPH=0, VECTOR={vector_score}，文档意图明确。"

    return None


# ----- V4.1 post-validation guards -----

_EXPLICIT_RELATION_INTENT_PATTERNS = (
    r"什么关系", r"是什么关系", r"有何关系",
    r"什么联系", r"是什么联系", r"有何联系",
    r"哪一种连接", r"哪种连接", r"关系标签", r"关系类型",
    r"哪个节点", r"哪一个节点", r"哪个实体", r"哪一个实体",
    r"上游实体", r"下游实体", r"谁指向", r"指向谁",
    r"连过来", r"连出去", r"怎么连接", r"如何连接",
    r"怎么关联", r"如何关联", r"怎么挂接", r"如何挂接",
    r"归到哪个", r"对应哪个", r"属于什么", r"谁导致", r"谁决定",
)

_KNOWLEDGE_QUERY_PATTERNS = (
    r"什么|为什么|如何|怎么|是否|会不会|能不能|多少|多大|多久|哪些|哪一|哪个|谁",
    r"关系|联系|影响|原因|机制|限制|定义|区别|包含|属于|对应",
    r"价格|成本|延迟|收入|用户数|日活|型号|配置|部署|云服务商|GPU",
)

_KB_OR_TECHNICAL_TERMS = (
    "上下文", "token", "agent", "多轮", "系统提示词", "工具定义", "工具调用",
    "推理过程", "模型", "旗舰模型", "长上下文", "成本", "延迟", "项目",
    "gpu", "部署", "云服务商", "用户", "收入", "知识库", "文档", "资料",
    "图谱", "neo4j",
)

def _has_explicit_relation_intent(question: str) -> bool:
    return _matches_any(_EXPLICIT_RELATION_INTENT_PATTERNS, question.strip().lower())

def _looks_like_knowledge_query(question: str) -> bool:
    q = question.strip().lower()
    return (
        _matches_any(_KNOWLEDGE_QUERY_PATTERNS, q)
        and any(term in q for term in _KB_OR_TECHNICAL_TERMS)
    )


# ----- V5 HYBRID boundary guard -----

def _validate_llm_route(question: str, route: str, reason: str) -> tuple[str, str]:
    q = question.strip()
    route = (route or "").strip().lower()
    graph_score, vector_score = _intent_scores(q)
    explicit_relation = _has_explicit_relation_intent(q)
    dual_evidence = _has_dual_evidence_intent(q)

    # If the user clearly asks for both evidence sources, never let the LLM
    # collapse the query to only VECTOR or only GRAPH.
    if dual_evidence and route != "hybrid":
        return (
            "hybrid",
            "V5双证据保护：用户明确要求文本证据与图谱证据共同回答。",
        )

    # HYBRID is not inferred merely because a relation can also be described in prose.
    if route == "hybrid" and not dual_evidence:
        if explicit_relation or graph_score > vector_score:
            return (
                "graph",
                f"V5 HYBRID边界保护：没有双证据请求，关系意图更强 "
                f"(GRAPH={graph_score}, VECTOR={vector_score})。",
            )
        if _looks_like_knowledge_query(q):
            return (
                "vector",
                f"V5 HYBRID边界保护：没有双证据请求，按知识事实查询处理 "
                f"(GRAPH={graph_score}, VECTOR={vector_score})。",
            )

    if route == "direct" and _looks_like_knowledge_query(q):
        if explicit_relation or graph_score > vector_score:
            return (
                "graph",
                f"V5 LLM结果保护：拒绝知识型DIRECT；关系意图更强 "
                f"(GRAPH={graph_score}, VECTOR={vector_score})。",
            )
        return (
            "vector",
            f"V5 LLM结果保护：拒绝知识型DIRECT；按文档知识查询处理 "
            f"(GRAPH={graph_score}, VECTOR={vector_score})。",
        )

    if route == "vector" and explicit_relation:
        return (
            "graph",
            f"V5 LLM结果保护：问题具有明确实体关系意图 "
            f"(GRAPH={graph_score}, VECTOR={vector_score})。",
        )

    if route == "graph" and not explicit_relation and vector_score >= 4 and graph_score == 0:
        return (
            "vector",
            f"V5 LLM结果保护：问题主要是事实/属性查询而非实体关系 "
            f"(GRAPH={graph_score}, VECTOR={vector_score})。",
        )

    return route, f"V5 LLM Router：{reason}"



def ingest_document(file_path: str, filename: str) -> dict:
    """
    文档入库：解析 -> 切块 -> 向量库 -> 高质量三元组抽取 -> 替换该文件旧图谱 -> 写入 Neo4j。

    关键点：
    1. 先完成三元组抽取，再删除旧图谱，避免在抽取还没开始时就把旧数据删掉。
    2. 同名文件重新上传时，删除/解除旧关系，再写入新版关系，避免图谱越堆越乱。
    3. 跨 chunk 的重复三元组在写库前再次去重。
    """
    text = parse_document(file_path)
    if not text.strip():
        raise ValueError("文档解析后内容为空，请检查文件是否损坏或是扫描版 PDF（暂不支持 OCR）")

    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    doc_id = str(uuid.uuid4())

    # 当前 vector_store_service 的 add_chunks 负责向量库入库/同名文档更新。
    add_chunks(doc_id=doc_id, filename=filename, chunks=chunks)

    all_triples: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        triples = extract_triples(chunk)
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            relation = str(triple.get("relation", "")).strip()
            obj = str(triple.get("object", "")).strip()

            if not subject or not relation or not obj:
                continue

            key = (subject.casefold(), relation.casefold(), obj.casefold())
            if key in seen:
                continue

            seen.add(key)
            all_triples.append({
                "subject": subject,
                "relation": relation,
                "object": obj,
            })

    # 三元组已经抽取完毕后，再替换该文件旧图谱。
    graph_cleanup = graph_service.delete_graph_by_source(filename)

    triple_count = graph_service.add_triples(
        all_triples,
        source_filename=filename,
    ) if all_triples else 0

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "triple_count": triple_count,
        "graph_cleanup": graph_cleanup,
    }

def _build_graph_context(question: str) -> str:
    """
    图谱检索：

    1. 通过 graph_service.match_entity_names() 做实体匹配；
       匹配已改成大小写不敏感，所以用户写 "Agent" 时，
       可以命中 Neo4j 中保存的 "agent"。
    2. 查询这些实体的一跳关系。
    3. 保留 Neo4j 的真实关系方向，拼成：
           source -[relation]-> target
       提供给最终回答模型。

    这样 GRAPH / HYBRID 路由就不会因为简单的大小写差异而误判为
    “没有图谱上下文”。
    """
    mentioned = graph_service.match_entity_names(question)

    if not mentioned:
        return ""

    relations = graph_service.find_related_entities(
        entity_names=mentioned,
        depth=1,
        limit=30,
    )

    if not relations:
        return ""

    lines = [
        f"{relation['source']} -[{relation['relation']}]-> {relation['target']}"
        for relation in relations
    ]

    return "\n".join(lines)


def _parse_router_response(content: str) -> tuple[str, str] | None:
    """解析 LLM Router 的 JSON 输出；兼容代码块或前后多余文字。"""
    if not content:
        return None

    cleaned = content.strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except Exception:
        return None

    route = str(data.get("route", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()

    if route not in {"direct", "vector", "graph", "hybrid"}:
        return None

    return route, reason or "由 LLM Router 根据问题意图选择。"


def _route_query(question: str) -> tuple[str, str]:
    """
    Agent Router V5:
    1. 显式模态约束
    2. 高置信度 DIRECT
    3. GRAPH / VECTOR 意图评分
    4. 不确定样本交给 LLM Router
    5. 对 LLM 结果做轻量语义保护
    6. API/解析失败才回退 VECTOR
    """
    q = question.strip()

    modality = _explicit_modality_route(q)
    if modality is not None:
        return modality

    if _is_direct_non_kb_task(q):
        return "direct", "V5高置信度非知识库任务命中，无需知识库检索。"

    scored = _high_confidence_intent_route(q)
    if scored is not None:
        return scored

    try:
        response = _llm_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=q)}],
            temperature=0,
        )
        parsed = _parse_router_response(response.choices[0].message.content)
        if parsed:
            route, reason = parsed
            return _validate_llm_route(q, route, reason)
    except Exception as e:
        print(f"Agent Router 判断失败，回退 VECTOR: {e}")

    return "vector", "V5 Router 调用失败，按知识库优先策略回退到文档向量检索。"



def _retrieve(question: str, route: str) -> tuple[list[dict], str]:
    """
    根据 Router 结果真正执行不同检索路线。

    DIRECT：不检索
    VECTOR：只查 Chroma
    GRAPH：只查 Neo4j
    HYBRID：Chroma + Neo4j
    """
    route = route.lower()

    if route == "direct":
        return [], ""

    if route == "vector":
        return search(question, top_k=settings.top_k), ""

    if route == "graph":
        return [], _build_graph_context(question)

    # hybrid
    hits = search(question, top_k=settings.top_k)
    graph_context = _build_graph_context(question)
    return hits, graph_context



def _actor_execute(
    question: str,
    route: str,
    attempt: int,
) -> tuple[list[dict], str, dict]:
    """
    Actor：执行 Router / Planner 已选定的检索动作。

    重要：这里内部仍然调用原有 _retrieve()，因此：
    - VECTOR 仍然只调用 Chroma；
    - GRAPH 仍然只调用 Neo4j；
    - HYBRID 仍然调用 Chroma + Neo4j；
    - 不改变 top_k、Embedding、图谱查询或检索排序逻辑。

    新增的只是结构化 Actor Trace，便于前端展示与面试解释。
    """
    hits, graph_context = _retrieve(question, route=route)

    trace = build_actor_trace(
        attempt=attempt,
        query=question,
        route=route,
        vector_hit_count=len(hits),
        graph_context_available=bool(graph_context),
    )

    return hits, graph_context, trace.to_dict()


def _definition_subject(question: str) -> str:
    """从高置信度定义问题中提取核心主语，用于 Context Judge 的轻量补充判断。"""
    q = re.sub(r"[？?！!。.\s]+$", "", question.strip())
    for suffix in ("是什么", "是什么意思", "指什么", "的定义", "定义是什么"):
        if suffix in q:
            subject = q.split(suffix, 1)[0].strip(" ：:，,")
            if 1 < len(subject) <= 50:
                return subject
    return ""


def _is_context_sufficient(
    hits: list[dict],
    graph_context: str,
    route: str,
    question: str = "",
) -> bool:
    """
    Route-aware Context Judge。

    - VECTOR：最高向量分数达到阈值；或者明确的“X 是什么”问题中，
      首个候选段落直接包含 X，视为高置信度命中。
    - GRAPH：必须检索到明确图谱关系。
    - HYBRID：两种来源都至少有结果，才认为混合上下文基本充分。
    - DIRECT：无需检索，视为充分。
    """
    route = route.lower()

    if route == "direct":
        return True

    if route == "graph":
        return bool(graph_context)

    if route == "vector":
        if hits and hits[0]["score"] >= settings.retrieval_sufficiency_threshold:
            return True

        subject = _definition_subject(question)
        if subject and hits:
            top_content = (hits[0].get("content") or "").lower()
            if subject.lower() in top_content:
                return True

        return False

    # hybrid：两种证据都存在即可进入生成；若缺一类则允许 Query Rewrite 尝试补全。
    return bool(hits) and bool(graph_context)


def _build_rewrite_evidence(
    hits: list[dict],
    graph_context: str,
    max_hits: int = 3,
    max_chars_per_hit: int = 420,
) -> str:
    """
    从第一次检索结果中构造 Query Rewrite 的“证据”。

    目的：
    - 让模型知道用户的模糊指代最可能对应知识库里的什么；
    - 同时严格限制模型只能在现有资料范围内消歧，减少语义漂移。

    只取前几个候选，并限制每段长度，避免为了改写查询塞入过多上下文。
    """
    parts = []

    for i, hit in enumerate(hits[:max_hits], start=1):
        content = (hit.get("content") or "").strip()
        content = re.sub(r"\s+", " ", content)
        if len(content) > max_chars_per_hit:
            content = content[:max_chars_per_hit] + "…"

        filename = hit.get("filename") or "未知文件"
        chunk_index = hit.get("chunk_index")
        segment = f"第{chunk_index + 1}段" if isinstance(chunk_index, int) else ""

        parts.append(
            f"[候选{i}] 来源：{filename} {segment}\n{content}"
        )

    if graph_context:
        graph_text = re.sub(r"\s+", " ", graph_context.strip())
        if len(graph_text) > 800:
            graph_text = graph_text[:800] + "…"
        parts.append(f"[知识图谱候选关系]\n{graph_text}")

    if not parts:
        return "（首次检索没有得到可用于消歧的资料；请只做最小改写，不要猜测具体实体。）"

    return "\n\n".join(parts)


def _clean_rewrite_output(text: str) -> str | None:
    """
    清理模型可能附带的引号、前缀和多余换行。
    Query Rewrite 最终只允许保留一条简洁查询。
    """
    if not text:
        return None

    cleaned = text.strip()

    prefixes = (
        "改写后的问题：",
        "改写后的查询：",
        "改写后的检索查询：",
        "查询：",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    cleaned = cleaned.strip("“”\"' ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 防止模型把查询扩写成很长的分析任务。
    if len(cleaned) > 220:
        cleaned = cleaned[:220].rstrip("，,；;：: ") + "？"

    return cleaned or None


def _rewrite_query(
    question: str,
    hits: list[dict],
    graph_context: str,
) -> str | None:
    """
    语义安全版 Query Rewrite。

    与旧版最大的区别：
    旧版只把原问题交给 LLM，因此“这种东西/它”等模糊指代容易被模型自行脑补。
    新版同时提供第一次检索得到的候选资料，并明确禁止引入资料之外的新概念。
    """
    evidence = _build_rewrite_evidence(hits, graph_context)

    try:
        response = _llm_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(
                        question=question,
                        evidence=evidence,
                    ),
                }
            ],
            # Query Rewrite 不是创作任务，温度越低越不容易发散。
            temperature=0.1,
        )

        rewritten = response.choices[0].message.content
        return _clean_rewrite_output(rewritten)
    except Exception as e:
        print(f"查询改写失败: {e}")
        return None


def _best_score(hits: list[dict]) -> float | None:
    """返回当前检索结果的最高分；没有结果时返回 None。"""
    if not hits:
        return None
    return float(hits[0]["score"])


def _should_adopt_rewrite(
    original_hits: list[dict],
    original_graph_context: str,
    rewritten_hits: list[dict],
    rewritten_graph_context: str,
    route: str,
    rewritten_question: str,
) -> bool:
    """
    判断同一路由下的二次检索是否值得采用。

    Query Rewrite 不改变 route：
    VECTOR 改写后仍查 VECTOR；
    GRAPH 改写后仍查 GRAPH；
    HYBRID 改写后仍查 HYBRID。
    """
    if _is_context_sufficient(
        rewritten_hits,
        rewritten_graph_context,
        route=route,
        question=rewritten_question,
    ):
        return True

    if route == "graph":
        return bool(rewritten_graph_context) and not bool(original_graph_context)

    if route == "vector":
        old_score = _best_score(original_hits)
        new_score = _best_score(rewritten_hits)
        return new_score is not None and (old_score is None or new_score > old_score)

    if route == "hybrid":
        # 混合检索中，如果改写后补齐了原来缺失的一类证据，则采用。
        old_modalities = int(bool(original_hits)) + int(bool(original_graph_context))
        new_modalities = int(bool(rewritten_hits)) + int(bool(rewritten_graph_context))
        if new_modalities > old_modalities:
            return True

        old_score = _best_score(original_hits)
        new_score = _best_score(rewritten_hits)
        if new_score is not None and (old_score is None or new_score > old_score):
            return True

    return False


def run_retrieval_pipeline(question: str) -> dict:
    """
    Planner -> Actor -> Reflector 显式化后的 Agentic RAG 检索管线。

        用户问题
            ↓
        Agent Router V5               （原逻辑不变）
            ↓
        Planner                       （新增：生成确定性执行计划）
            ↓
        Actor                         （新增显式层：执行原 _retrieve）
            ↓
        Reflector / Context Judge     （复用原 _is_context_sufficient）
            ↓
      ┌─────┴──────────────┐
      │充分                │不足
      ↓                    ↓
    Answer            Query Rewrite
                           ↓
                       Actor Retry
                           ↓
                       Reflector

    这一改造只增加编排与可观察信息，不改变：
    - Router V5；
    - Chroma / Neo4j 检索；
    - Context Judge 判定规则；
    - Query Rewrite 的采用规则。
    """
    route, route_reason = _route_query(question)

    # Planner 不额外调用 LLM，只根据已经确定的 route 展开执行计划。
    plan = build_plan(route, route_reason=route_reason)

    actor_trace: list[dict] = []
    reflection_history: list[dict] = []

    if route == "direct":
        reflection = build_reflection(
            round_index=0,
            route=route,
            sufficient=True,
            rewrite_allowed=False,
            vector_hit_count=0,
            graph_context_available=False,
        ).to_dict()
        reflection_history.append(reflection)

        return {
            "route": route,
            "route_reason": route_reason,
            "hits": [],
            "graph_context": "",
            "rewrite_attempted": False,
            "rewrite_candidate": None,
            "rewritten_query": None,
            "skipped_retrieval": True,
            "context_sufficient": True,
            "planner": plan.to_dict(),
            "actor_trace": actor_trace,
            "reflection": reflection,
            "reflection_history": reflection_history,
        }

    # ------------------------------------------------------------------
    # Actor round 1: 完全复用原来的 _retrieve(question, route)。
    # ------------------------------------------------------------------
    hits, graph_context, trace1 = _actor_execute(
        question=question,
        route=route,
        attempt=1,
    )
    actor_trace.append(trace1)

    first_sufficient = _is_context_sufficient(
        hits,
        graph_context,
        route=route,
        question=question,
    )

    first_reflection = build_reflection(
        round_index=1,
        route=route,
        sufficient=first_sufficient,
        rewrite_allowed=not first_sufficient,
        vector_hit_count=len(hits),
        graph_context_available=bool(graph_context),
    ).to_dict()
    reflection_history.append(first_reflection)

    rewrite_attempted = False
    rewrite_candidate = None
    rewritten_query = None
    second_actor_executed = False

    # ------------------------------------------------------------------
    # Reflector 判断不足时，沿用原有 Query Rewrite + 原 route 重试逻辑。
    # ------------------------------------------------------------------
    if not first_sufficient:
        rewrite_attempted = True

        candidate_query = _rewrite_query(
            question=question,
            hits=hits,
            graph_context=graph_context,
        )
        rewrite_candidate = candidate_query

        if candidate_query and candidate_query.strip() != question.strip():
            hits2, graph_context2, trace2 = _actor_execute(
                question=candidate_query,
                route=route,
                attempt=2,
            )
            actor_trace.append(trace2)
            second_actor_executed = True

            # 重要：仍然使用原 _should_adopt_rewrite()，不改变采用标准。
            if _should_adopt_rewrite(
                original_hits=hits,
                original_graph_context=graph_context,
                rewritten_hits=hits2,
                rewritten_graph_context=graph_context2,
                route=route,
                rewritten_question=candidate_query,
            ):
                hits, graph_context = hits2, graph_context2
                rewritten_query = candidate_query

    # 最终 Reflector 只描述“最终被采用的上下文”，不修改任何检索结果。
    final_question = rewritten_query or question
    final_sufficient = _is_context_sufficient(
        hits,
        graph_context,
        route=route,
        question=final_question,
    )

    # 第一次已经充分时，first_reflection 就是最终结果，不重复记录。
    # 如果尝试了 Rewrite，则追加一个最终反思，明确闭环结束状态。
    if rewrite_attempted:
        final_round = 2 if second_actor_executed else 1
        final_reflection = build_reflection(
            round_index=final_round,
            route=route,
            sufficient=final_sufficient,
            rewrite_allowed=False,
            vector_hit_count=len(hits),
            graph_context_available=bool(graph_context),
        ).to_dict()
        reflection_history.append(final_reflection)
    else:
        final_reflection = first_reflection

    return {
        "route": route,
        "route_reason": route_reason,
        "hits": hits,
        "graph_context": graph_context,
        "rewrite_attempted": rewrite_attempted,
        "rewrite_candidate": rewrite_candidate,
        "rewritten_query": rewritten_query,
        "skipped_retrieval": False,
        "context_sufficient": final_sufficient,
        "planner": plan.to_dict(),
        "actor_trace": actor_trace,
        "reflection": final_reflection,
        "reflection_history": reflection_history,
    }


def _build_prompt(question: str, hits: list[dict], graph_context: str) -> str:
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        # chunk_index 从 0 开始存储，展示时 +1 更符合"第几段"的自然计数习惯
        segment_label = f"第{hit['chunk_index'] + 1}段" if hit.get("chunk_index") is not None else ""
        context_parts.append(f"[资料{i}] 来源：{hit['filename']} {segment_label}\n{hit['content']}")
    context_text = "\n\n".join(context_parts) if context_parts else "（无相关文档片段）"

    graph_section = f"\n\n【知识图谱关联信息】\n{graph_context}" if graph_context else ""

    return f"""【参考资料】
{context_text}{graph_section}

【用户问题】
{question}
"""


def _sources_from_hits(hits: list[dict]) -> list[dict]:
    return [
        {"filename": hit["filename"], "chunk_index": hit.get("chunk_index"), "score": round(hit["score"], 3)}
        for hit in hits
    ]


def _direct_answer(question: str) -> str:
    """DIRECT 路由的非流式回答，不访问知识库。"""
    response = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _direct_answer_stream(question: str):
    """DIRECT 路由的流式回答，不访问知识库。"""
    stream = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta



def basic_answer_question(question: str) -> dict:
    """Basic RAG：只执行一次 Chroma 向量检索，然后生成回答。"""
    hits = search(question, top_k=settings.top_k)

    if not hits:
        return {
            "answer": "知识库中没有检索到相关内容，请先确认已上传对应文档。",
            "sources": [],
            "route": "basic_rag",
            "route_reason": "Basic RAG 固定执行一次 Chroma 向量检索，不经过 Agent Router。",
        }

    user_prompt = _build_prompt(question, hits, "")
    response = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": BASIC_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": _sources_from_hits(hits),
        "route": "basic_rag",
        "route_reason": "Basic RAG 固定执行一次 Chroma 向量检索，不经过 Agent Router。",
    }


def stream_basic_answer(question: str):
    """
    Basic RAG 流式回答。

    流程固定为：
        Question -> Chroma Top-K -> LLM stream

    不经过 Router / Planner / Actor / Reflector / Query Rewrite / Neo4j。
    """
    hits = search(question, top_k=settings.top_k)

    yield {
        "type": "basic_info",
        "data": {
            "mode": "basic_rag",
            "vector_hit_count": len(hits),
        },
    }

    if not hits:
        yield {"type": "sources", "data": []}
        yield {
            "type": "content",
            "data": "知识库中没有检索到相关内容，请先确认已上传对应文档。",
        }
        return

    user_prompt = _build_prompt(question, hits, "")
    yield {"type": "sources", "data": _sources_from_hits(hits)}

    stream = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": BASIC_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "content", "data": delta}

def answer_question(question: str) -> dict:
    """非流式问答：Router -> Planner -> Actor -> Reflector -> Answer。"""
    pipeline_result = run_retrieval_pipeline(question)
    route = pipeline_result["route"]
    route_reason = pipeline_result["route_reason"]
    hits = pipeline_result["hits"]
    graph_context = pipeline_result["graph_context"]

    common_trace = {
        "route": route,
        "route_reason": route_reason,
        "rewrite_attempted": pipeline_result["rewrite_attempted"],
        "rewrite_candidate": pipeline_result["rewrite_candidate"],
        "rewritten_query": pipeline_result["rewritten_query"],
        "context_sufficient": pipeline_result["context_sufficient"],
        "planner": pipeline_result["planner"],
        "actor_trace": pipeline_result["actor_trace"],
        "reflection": pipeline_result["reflection"],
        "reflection_history": pipeline_result["reflection_history"],
    }

    if route == "direct":
        return {
            "answer": _direct_answer(question),
            "sources": [],
            **common_trace,
        }

    if not hits and not graph_context:
        return {
            "answer": "知识库中没有检索到足够的相关内容，请先确认文档或知识图谱中存在对应信息。",
            "sources": [],
            **common_trace,
        }

    user_prompt = _build_prompt(question, hits, graph_context)

    response = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": _sources_from_hits(hits),
        **common_trace,
    }


def stream_answer(question: str):
    """
    流式版本。

    新增可观察事件：
    - planner：Planner 生成的确定性计划
    - actor：每一轮真实检索执行
    - reflector：Context Judge / Reflector 的判断

    原有事件保持不变：
    - retrieval_info
    - sources
    - content
    """
    pipeline_result = run_retrieval_pipeline(question)
    route = pipeline_result["route"]
    route_reason = pipeline_result["route_reason"]
    hits = pipeline_result["hits"]
    graph_context = pipeline_result["graph_context"]

    # 先把完整 Agent Trace 交给前端。
    yield {
        "type": "planner",
        "data": pipeline_result["planner"],
    }

    for actor_item in pipeline_result["actor_trace"]:
        yield {
            "type": "actor",
            "data": actor_item,
        }

    for reflection_item in pipeline_result["reflection_history"]:
        yield {
            "type": "reflector",
            "data": reflection_item,
        }

    yield {
        "type": "retrieval_info",
        "data": {
            "route": route,
            "route_reason": route_reason,
            "rewrite_attempted": pipeline_result["rewrite_attempted"],
            "rewrite_candidate": pipeline_result["rewrite_candidate"],
            "rewritten_query": pipeline_result["rewritten_query"],
            "skipped_retrieval": pipeline_result["skipped_retrieval"],
            "context_sufficient": pipeline_result["context_sufficient"],
            # 这里也冗余附带一次，方便旧前端或只监听 retrieval_info 的客户端兼容。
            "planner": pipeline_result["planner"],
            "actor_trace": pipeline_result["actor_trace"],
            "reflection": pipeline_result["reflection"],
            "reflection_history": pipeline_result["reflection_history"],
        },
    }

    if route == "direct":
        yield {"type": "sources", "data": []}
        for delta in _direct_answer_stream(question):
            yield {"type": "content", "data": delta}
        return

    if not hits and not graph_context:
        yield {"type": "sources", "data": []}
        yield {
            "type": "content",
            "data": "知识库中没有检索到足够的相关内容，请先确认文档或知识图谱中存在对应信息。",
        }
        return

    user_prompt = _build_prompt(question, hits, graph_context)
    yield {"type": "sources", "data": _sources_from_hits(hits)}

    stream = _llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "content", "data": delta}
