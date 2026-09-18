"""
Planner / Actor / Reflector 显式编排层。

设计原则：
1. 不替换现有 Router V5，不改变 _route_query() 的路由结果。
2. 不替换 Chroma / Neo4j 检索，不改变检索算法和参数。
3. 不额外调用 LLM 生成计划，Planner 使用确定性计划，避免增加延迟。
4. Actor 只记录和描述现有检索动作。
5. Reflector 复用现有 Context Judge 的充分性判断结果。

因此，这一层的作用是“显式化 Agent 架构 + 增强可观察性”，
而不是修改已经完成评测的 Router / Retrieval 核心算法。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_ROUTES = {"direct", "vector", "graph", "hybrid"}


@dataclass(frozen=True)
class PlanStep:
    """Planner 生成的一步执行计划。"""

    key: str
    label: str
    action: str
    evidence_source: str | None = None
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentPlan:
    """一次用户请求对应的确定性执行计划。"""

    route: str
    reason: str
    steps: tuple[PlanStep, ...]
    max_retrieval_rounds: int = 2

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reason": self.reason,
            "max_retrieval_rounds": self.max_retrieval_rounds,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class ActorTrace:
    """Actor 一次真实执行的追踪信息。"""

    attempt: int
    query: str
    route: str
    actions: tuple[str, ...]
    vector_hit_count: int
    graph_context_available: bool

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "query": self.query,
            "route": self.route,
            "actions": list(self.actions),
            "vector_hit_count": self.vector_hit_count,
            "graph_context_available": self.graph_context_available,
        }


@dataclass(frozen=True)
class ReflectionResult:
    """Reflector 对当前证据状态的结构化判断。"""

    round_index: int
    route: str
    sufficient: bool
    action: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_route(route: str) -> str:
    value = (route or "").strip().lower()
    return value if value in VALID_ROUTES else "vector"


def build_plan(route: str, route_reason: str = "") -> AgentPlan:
    """
    Planner：根据 Router 已经选定的 route 生成确定性执行计划。

    注意：这里故意不再次调用 LLM。
    Router 已经负责“应该走哪条路线”，Planner 只负责把该路线展开成
    可解释、可展示的步骤。
    """
    route = _normalize_route(route)

    if route == "direct":
        steps = (
            PlanStep(
                key="direct_answer",
                label="Direct Answer",
                action="direct_answer",
                evidence_source=None,
            ),
        )

    elif route == "vector":
        steps = (
            PlanStep(
                key="vector_retrieval",
                label="Vector Retrieval",
                action="vector_retrieval",
                evidence_source="chroma",
            ),
            PlanStep(
                key="reflect",
                label="Reflector / Context Judge",
                action="context_judge",
            ),
            PlanStep(
                key="answer_generation",
                label="Answer Generation",
                action="answer_generation",
            ),
        )

    elif route == "graph":
        steps = (
            PlanStep(
                key="graph_retrieval",
                label="Graph Retrieval",
                action="graph_retrieval",
                evidence_source="neo4j",
            ),
            PlanStep(
                key="reflect",
                label="Reflector / Context Judge",
                action="context_judge",
            ),
            PlanStep(
                key="answer_generation",
                label="Answer Generation",
                action="answer_generation",
            ),
        )

    else:  # hybrid
        steps = (
            PlanStep(
                key="vector_retrieval",
                label="Vector Retrieval",
                action="vector_retrieval",
                evidence_source="chroma",
            ),
            PlanStep(
                key="graph_retrieval",
                label="Graph Retrieval",
                action="graph_retrieval",
                evidence_source="neo4j",
            ),
            PlanStep(
                key="reflect",
                label="Reflector / Context Judge",
                action="context_judge",
            ),
            PlanStep(
                key="answer_generation",
                label="Answer Generation",
                action="answer_generation",
            ),
        )

    return AgentPlan(
        route=route,
        reason=route_reason or f"根据 Router 结果生成 {route.upper()} 执行计划。",
        steps=steps,
        max_retrieval_rounds=2,
    )


def actor_actions_for_route(route: str) -> tuple[str, ...]:
    """把 route 映射为 Actor 真正执行的动作。"""
    route = _normalize_route(route)

    if route == "direct":
        return ("direct_answer",)
    if route == "vector":
        return ("vector_retrieval",)
    if route == "graph":
        return ("graph_retrieval",)
    return ("vector_retrieval", "graph_retrieval")


def build_actor_trace(
    *,
    attempt: int,
    query: str,
    route: str,
    vector_hit_count: int,
    graph_context_available: bool,
) -> ActorTrace:
    """把一次真实检索执行转成 Actor Trace。"""
    route = _normalize_route(route)
    return ActorTrace(
        attempt=max(1, int(attempt)),
        query=query,
        route=route,
        actions=actor_actions_for_route(route),
        vector_hit_count=max(0, int(vector_hit_count)),
        graph_context_available=bool(graph_context_available),
    )


def build_reflection(
    *,
    round_index: int,
    route: str,
    sufficient: bool,
    rewrite_allowed: bool,
    vector_hit_count: int = 0,
    graph_context_available: bool = False,
) -> ReflectionResult:
    """
    Reflector：把现有 Context Judge 的 bool 结果显式化。

    本函数不重新判断证据，也不调用 LLM，只负责：
    - 记录“够 / 不够”；
    - 选择下一步 action；
    - 生成可解释 reason。
    """
    route = _normalize_route(route)
    sufficient = bool(sufficient)

    if route == "direct":
        return ReflectionResult(
            round_index=round_index,
            route=route,
            sufficient=True,
            action="direct_answer",
            reason="DIRECT 路线无需知识库证据，直接进入回答生成。",
        )

    evidence_summary = (
        f"vector_hits={max(0, int(vector_hit_count))}, "
        f"graph_context={'yes' if graph_context_available else 'no'}"
    )

    if sufficient:
        return ReflectionResult(
            round_index=round_index,
            route=route,
            sufficient=True,
            action="answer",
            reason=(
                f"Context Judge 判断 {route.upper()} 路线证据充分 "
                f"({evidence_summary})，进入回答生成。"
            ),
        )

    if rewrite_allowed:
        return ReflectionResult(
            round_index=round_index,
            route=route,
            sufficient=False,
            action="rewrite_and_retry",
            reason=(
                f"Context Judge 判断 {route.upper()} 路线证据不足 "
                f"({evidence_summary})，触发 Query Rewrite，并保持原 route 重试。"
            ),
        )

    return ReflectionResult(
        round_index=round_index,
        route=route,
        sufficient=False,
        action="answer_or_safe_refusal",
        reason=(
            f"达到最大检索轮次后证据仍不足 ({evidence_summary})，"
            "不再继续循环；由最终回答阶段基于现有证据回答或安全拒答。"
        ),
    )
