from app.services.rag_service import _route_query

CASES = [
    ("hybrid", "先用项目文档说明 Actor 的作用，并沿知识图谱查看它后续连接到哪些节点。"),
    ("hybrid", "文字证据和图谱关系两边都要：解释 Vector Store 的职责以及它和 Embedding Service 的连接。"),
    ("hybrid", "别只给文档定义，也别只列图谱节点，请同时说明 Graph Retrieval 的功能和后续关系链。"),
    ("hybrid", "一方面根据说明材料解释 Context Judge 的作用，另一方面根据关系图说明它后面接什么节点。"),
    ("hybrid", "从项目资料说明安全拒答的目的，同时从流程关系定位它在回答链路中的位置。"),
    ("hybrid", "把说明文档中 top_k 的含义与图谱中 Vector Retrieval 周边关系合到一个答案里。"),
    ("hybrid", "用 README 说明 FastAPI Backend 的职责，并从系统图补充它连接的组件。"),
    ("hybrid", "先确认文档中 SSE 的用途，然后从组件关系图说明事件最终流向哪个前端模块。"),
    ("graph", "安全拒答在最终回答链路中位于什么位置？"),
    ("graph", "Context Judge 判断证据不足以后，流程接下来进入哪里？"),
    ("graph", "Reflector 完成判断后下一步进入哪个环节？"),
    ("graph", "Query Rewrite 之后流程会回到哪一个执行模块？"),
    ("vector", "文档中如何说明 Actor 的主要职责？"),
    ("vector", "为什么 Context Judge 要检查证据是否充分？"),
    ("vector", "README 是否写了 FastAPI Backend 的启动方式？"),
    ("vector", "SSE 在项目中的主要用途是什么？"),
    ("graph", "Actor 和 Reflector 在流程中是什么关系？"),
    ("graph", "Embedding Service 的输出连接到哪个存储节点？"),
    ("direct", "给 README 写一句很短的欢迎语。"),
    ("direct", "把“GRAPH ROUTER”全部改成小写。"),
]

def main():
    passed = 0
    failed = []
    for i, (expected, question) in enumerate(CASES, start=1):
        predicted, reason = _route_query(question)
        ok = predicted == expected
        status = "PASS" if ok else "FAIL"
        print(f"[{i:02d}/{len(CASES)}] {status} expected={expected.upper():6s} predicted={predicted.upper():6s}")
        print(f"  Q: {question}")
        print(f"  reason: {reason}")
        if ok:
            passed += 1
        else:
            failed.append((i, expected, predicted, question, reason))
    print("\n" + "=" * 72)
    print(f"V5.2 boundary dev test: {passed}/{len(CASES)} PASS")
    if failed:
        print("Failures:")
        for i, expected, predicted, question, reason in failed:
            print(f"- id={i} expected={expected} predicted={predicted}")
            print(f"  {question}")
            print(f"  {reason}")
    print("=" * 72)
    raise SystemExit(0 if passed == len(CASES) else 1)

if __name__ == "__main__":
    main()
