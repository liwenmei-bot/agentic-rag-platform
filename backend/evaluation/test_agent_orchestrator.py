"""Planner / Actor / Reflector standalone smoke test.

This test does not access Chroma, Neo4j or LLM API.
Run from backend root:
    python -m evaluation.test_agent_orchestrator
"""

from app.services.agent_orchestrator import (
    actor_actions_for_route,
    build_actor_trace,
    build_plan,
    build_reflection,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    routes = ["direct", "vector", "graph", "hybrid"]

    print("=" * 72)
    print("Planner / Actor / Reflector Smoke Test")
    print("=" * 72)

    for route in routes:
        plan = build_plan(route, f"test-{route}")
        actions = actor_actions_for_route(route)

        check(plan.route == route, f"plan route mismatch: {route}")
        check(len(plan.steps) >= 1, f"plan steps missing: {route}")
        check(len(actions) >= 1, f"actor actions missing: {route}")

        print(f"[PLAN] {route.upper():<6} -> {[step.action for step in plan.steps]}")
        print(f"[ACT ] {route.upper():<6} -> {list(actions)}")

    vector_trace = build_actor_trace(
        attempt=1,
        query="什么是 Chroma？",
        route="vector",
        vector_hit_count=5,
        graph_context_available=False,
    )
    check(vector_trace.actions == ("vector_retrieval",), "VECTOR actor mismatch")

    hybrid_trace = build_actor_trace(
        attempt=1,
        query="结合文档和知识图谱解释 Router",
        route="hybrid",
        vector_hit_count=5,
        graph_context_available=True,
    )
    check(
        hybrid_trace.actions == ("vector_retrieval", "graph_retrieval"),
        "HYBRID actor mismatch",
    )

    insufficient = build_reflection(
        round_index=1,
        route="vector",
        sufficient=False,
        rewrite_allowed=True,
        vector_hit_count=5,
        graph_context_available=False,
    )
    check(insufficient.action == "rewrite_and_retry", "Reflector retry mismatch")

    sufficient = build_reflection(
        round_index=1,
        route="graph",
        sufficient=True,
        rewrite_allowed=False,
        vector_hit_count=0,
        graph_context_available=True,
    )
    check(sufficient.action == "answer", "Reflector answer mismatch")

    final_insufficient = build_reflection(
        round_index=2,
        route="hybrid",
        sufficient=False,
        rewrite_allowed=False,
        vector_hit_count=3,
        graph_context_available=False,
    )
    check(
        final_insufficient.action == "answer_or_safe_refusal",
        "Reflector final fallback mismatch",
    )

    print()
    print("[REF] insufficient ->", insufficient.action)
    print("[REF] sufficient   ->", sufficient.action)
    print("[REF] final weak   ->", final_insufficient.action)
    print()
    print("Result: PASS")


if __name__ == "__main__":
    main()
