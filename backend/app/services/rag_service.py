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

ROUTER_PROMPT = """你是 Agentic RAG 的路由器。请只判断当前问题应该走哪条路线。

可选路线只有：
- direct：寒暄、感谢、告别，或明显不需要知识库的简单通用对话。
- vector：主要需要从文档中查定义、说明、段落内容、事实描述。
- graph：主要询问两个或多个实体之间的明确关系、连接、依赖、归属。
- hybrid：同时需要文档语义内容和实体关系，或问题较复杂，需要两类证据共同回答。

判断规则：
1. “是什么/定义/文档里讲什么”优先 vector。
2. “A 和 B 是什么关系/如何关联”优先 graph。
3. 明确要求“结合文档和知识图谱”“结合知识库和图谱”必须 hybrid。
4. 不要因为问题里出现 Agent、上下文、token 等词就自动选 graph。
5. 如果只是“你好/谢谢/再见”等，选 direct。
6. 无法确定时，优先 vector；只有确实需要两种检索方式时才选 hybrid。

只输出一个 JSON 对象，不要 Markdown，不要额外解释：
{{"route":"vector","reason":"问题主要询问文档中的概念定义"}}

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

# 闲聊/无需检索的问题的简单特征词，命中且问题很短时直接跳过检索，节省一次检索+LLM调用。
# 这是启发式判断（V1），更严谨的做法是用一次轻量 LLM 调用做意图分类，
# 但对"你好""谢谢"这类高置信度场景，用规则判断延迟更低、成本更低。
_CHITCHAT_PATTERNS = re.compile(
    r"^(你好|您好|hi|hello|嗨|谢谢|感谢|thanks|thank you|再见|拜拜|bye)[!！。.，,\s]*$",
    re.IGNORECASE,
)


def _is_chitchat(question: str) -> bool:
    return bool(_CHITCHAT_PATTERNS.match(question.strip())) and len(question.strip()) <= 10


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
    Agent Router V1：规则优先 + LLM 兜底。

    高置信度场景使用规则，减少一次 LLM 调用；
    规则无法稳定判断时，再让 LLM 在 DIRECT / VECTOR / GRAPH / HYBRID 中四选一。
    """
    q = question.strip()
    q_lower = q.lower()

    if _is_chitchat(q):
        return "direct", "属于高置信度寒暄/感谢/告别，不需要知识库检索。"

    # 明确要求同时使用文档和知识图谱时，直接 HYBRID。
    has_doc_word = any(word in q for word in ("文档", "资料", "知识库"))
    has_graph_word = any(word in q for word in ("知识图谱", "图谱", "Neo4j", "neo4j"))
    if (has_doc_word and has_graph_word) or any(
        phrase in q
        for phrase in (
            "结合文档和知识图谱",
            "结合知识库和知识图谱",
            "结合文档与知识图谱",
            "结合知识库与知识图谱",
        )
    ):
        return "hybrid", "问题明确要求同时结合文档内容和知识图谱关系。"

    # 明确关系型问题优先 GRAPH。
    relation_patterns = (
        r"什么关系",
        r"有何关系",
        r"关系是什么",
        r"之间.*关系",
        r"如何关联",
        r"怎么关联",
        r"关联关系",
    )
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in relation_patterns):
        return "graph", "问题主要询问实体之间的明确关系，适合知识图谱检索。"

    # 明确的定义/文档内容问题优先 VECTOR。
    vector_patterns = (
        r"是什么",
        r"什么意思",
        r"定义",
        r"文档里",
        r"文档中",
        r"资料里",
        r"资料中",
        r"知识库里",
        r"知识库中",
        r"讲了什么",
        r"介绍",
        r"说明",
    )
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in vector_patterns):
        return "vector", "问题主要询问文档中的定义、说明或文本内容。"

    # 其余问题交给轻量 Router LLM。
    try:
        response = _llm_client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {
                    "role": "user",
                    "content": ROUTER_PROMPT.format(question=q),
                }
            ],
            temperature=0,
        )
        parsed = _parse_router_response(response.choices[0].message.content)
        if parsed:
            return parsed
    except Exception as e:
        print(f"Agent Router 判断失败，回退 VECTOR: {e}")

    return "vector", "规则和 LLM Router 未能稳定分类，按保守策略回退到文档向量检索。"


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
    真正的 Agent Router 检索管线：

        用户问题
            ↓
        Agent Router
            ↓
    DIRECT / VECTOR / GRAPH / HYBRID
            ↓
      对应检索方式
            ↓
      Context Judge
            ↓
      必要时 Query Rewrite
            ↓
      按原 route 二次检索
    """
    route, route_reason = _route_query(question)

    if route == "direct":
        return {
            "route": route,
            "route_reason": route_reason,
            "hits": [],
            "graph_context": "",
            "rewrite_attempted": False,
            "rewrite_candidate": None,
            "rewritten_query": None,
            "skipped_retrieval": True,
        }

    hits, graph_context = _retrieve(question, route=route)

    rewrite_attempted = False
    rewrite_candidate = None
    rewritten_query = None

    if not _is_context_sufficient(
        hits,
        graph_context,
        route=route,
        question=question,
    ):
        rewrite_attempted = True

        candidate_query = _rewrite_query(
            question=question,
            hits=hits,
            graph_context=graph_context,
        )
        rewrite_candidate = candidate_query

        if candidate_query and candidate_query.strip() != question.strip():
            # 重要：改写查询不改变 Router 决策，只在原 route 内重试。
            hits2, graph_context2 = _retrieve(candidate_query, route=route)

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

    return {
        "route": route,
        "route_reason": route_reason,
        "hits": hits,
        "graph_context": graph_context,
        "rewrite_attempted": rewrite_attempted,
        "rewrite_candidate": rewrite_candidate,
        "rewritten_query": rewritten_query,
        "skipped_retrieval": False,
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


def answer_question(question: str) -> dict:
    """非流式问答：Agent Router -> 检索/直答 -> Context Judge -> Query Rewrite。"""
    pipeline_result = run_retrieval_pipeline(question)
    route = pipeline_result["route"]
    route_reason = pipeline_result["route_reason"]
    hits = pipeline_result["hits"]
    graph_context = pipeline_result["graph_context"]

    if route == "direct":
        return {
            "answer": _direct_answer(question),
            "sources": [],
            "route": route,
            "route_reason": route_reason,
            "rewrite_attempted": False,
            "rewrite_candidate": None,
            "rewritten_query": None,
        }

    if not hits and not graph_context:
        return {
            "answer": "知识库中没有检索到足够的相关内容，请先确认文档或知识图谱中存在对应信息。",
            "sources": [],
            "route": route,
            "route_reason": route_reason,
            "rewrite_attempted": pipeline_result["rewrite_attempted"],
            "rewrite_candidate": pipeline_result["rewrite_candidate"],
            "rewritten_query": pipeline_result["rewritten_query"],
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
        "route": route,
        "route_reason": route_reason,
        "rewrite_attempted": pipeline_result["rewrite_attempted"],
        "rewrite_candidate": pipeline_result["rewrite_candidate"],
        "rewritten_query": pipeline_result["rewritten_query"],
    }


def stream_answer(question: str):
    """
    流式版本。

    retrieval_info 会把 Router、Context Judge、Query Rewrite 的决策一并返回前端。
    """
    pipeline_result = run_retrieval_pipeline(question)
    route = pipeline_result["route"]
    route_reason = pipeline_result["route_reason"]
    hits = pipeline_result["hits"]
    graph_context = pipeline_result["graph_context"]

    yield {
        "type": "retrieval_info",
        "data": {
            "route": route,
            "route_reason": route_reason,
            "rewrite_attempted": pipeline_result["rewrite_attempted"],
            "rewrite_candidate": pipeline_result["rewrite_candidate"],
            "rewritten_query": pipeline_result["rewritten_query"],
            "skipped_retrieval": pipeline_result["skipped_retrieval"],
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

