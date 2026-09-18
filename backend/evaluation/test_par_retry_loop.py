from app.services import rag_service


def main():
    original_retrieve = rag_service._retrieve
    original_context_judge = rag_service._is_context_sufficient
    original_rewrite = rag_service._rewrite_query
    original_adopt = rag_service._should_adopt_rewrite
    original_route = rag_service._route_query

    calls = {
        "retrieve": 0,
        "judge": 0,
        "rewrite": 0,
    }

    try:
        # 固定 VECTOR，避免测试受 Router 变化影响
        rag_service._route_query = lambda question: (
            "vector",
            "PAR retry smoke test",
        )

        def fake_retrieve(question, route):
            calls["retrieve"] += 1

            if calls["retrieve"] == 1:
                return (
                    [
                        {
                            "content": "第一次检索得到的证据不充分。",
                            "filename": "test.txt",
                            "doc_id": "test-doc",
                            "chunk_index": 0,
                            "score": 0.10,
                        }
                    ],
                    "",
                )

            return (
                [
                    {
                        "content": "第二次检索得到充分证据。",
                        "filename": "test.txt",
                        "doc_id": "test-doc",
                        "chunk_index": 1,
                        "score": 0.90,
                    }
                ],
                "",
            )

        def fake_context_judge(
            hits,
            graph_context,
            route,
            question="",
        ):
            calls["judge"] += 1

            # 第一轮不足，第二轮充分
            return calls["judge"] >= 2

        def fake_rewrite(
            question,
            hits,
            graph_context,
        ):
            calls["rewrite"] += 1
            return "改写后的测试查询"

        rag_service._retrieve = fake_retrieve
        rag_service._is_context_sufficient = fake_context_judge
        rag_service._rewrite_query = fake_rewrite
        rag_service._should_adopt_rewrite = (
            lambda **kwargs: True
        )

        result = rag_service.run_retrieval_pipeline(
            "原始测试问题"
        )

        print("=" * 70)
        print("Planner / Actor / Reflector Retry Loop Test")
        print("=" * 70)

        print("route:", result["route"])
        print("rewrite_attempted:", result["rewrite_attempted"])
        print("rewrite_candidate:", result["rewrite_candidate"])
        print("rewritten_query:", result["rewritten_query"])

        print("\nActor Trace:")
        for item in result["actor_trace"]:
            print(item)

        print("\nReflection History:")
        for item in result["reflection_history"]:
            print(item)

        print("\nCall counts:")
        print(calls)

        assert result["route"] == "vector"
        assert result["rewrite_attempted"] is True
        assert result["rewritten_query"] == "改写后的测试查询"

        assert len(result["actor_trace"]) == 2
        assert result["actor_trace"][0]["attempt"] == 1
        assert result["actor_trace"][1]["attempt"] == 2

        assert len(result["reflection_history"]) == 2
        assert result["reflection_history"][0]["sufficient"] is False
        assert result["reflection_history"][1]["sufficient"] is True

        assert calls["retrieve"] == 2
        assert calls["rewrite"] == 1

        print("\nResult: PASS")

    finally:
        # 恢复正式函数，保证测试不会污染运行环境
        rag_service._retrieve = original_retrieve
        rag_service._is_context_sufficient = original_context_judge
        rag_service._rewrite_query = original_rewrite
        rag_service._should_adopt_rewrite = original_adopt
        rag_service._route_query = original_route


if __name__ == "__main__":
    main()