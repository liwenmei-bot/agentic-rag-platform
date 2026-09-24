import json
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.services import agent_service, memory_service, rag_service, session_service
from app.services.agent_tools import ToolError, execute_tool
from evaluation.evaluate_answers import score_case, summarize


def test_pipeline_events_precede_retrieval_and_preserve_nonstream_result(monkeypatch):
    calls = []
    monkeypatch.setattr(rag_service, "_route_query", lambda _: ("vector", "test route"))
    monkeypatch.setattr(rag_service, "_actor_execute", lambda **kw: (
        calls.append(kw["attempt"]) or
        ([{"filename": "doc.txt", "chunk_index": 0, "score": 0.8, "content": "证据"}],
         "", {"attempt": kw["attempt"], "actions": ["vector_retrieval"]})
    ))
    monkeypatch.setattr(rag_service, "_is_context_sufficient", lambda *a, **kw: True)

    stream = rag_service.stream_retrieval_pipeline("问题")
    assert next(stream)["data"]["stage"] == "router"
    assert calls == []
    assert next(stream)["type"] == "router"
    assert next(stream)["type"] == "planner"
    assert next(stream)["data"]["stage"] == "actor"
    assert calls == []  # Client can render a live Actor status before search runs.
    assert next(stream)["type"] == "actor"
    assert calls == [1]
    events = list(stream)
    assert any(e["type"] == "reflector" for e in events)
    result = rag_service.run_retrieval_pipeline("问题")
    assert result["route"] == "vector"
    assert result["context_sufficient"] is True
    assert calls == [1, 1]


def test_stream_writes_only_completed_turns_and_isolates_sessions(monkeypatch, tmp_path):
    monkeypatch.setattr(session_service, "DB_PATH", tmp_path / "app.db")
    from app.api import chat
    monkeypatch.setattr(chat, "resolve_question", lambda question, history: (
        "关于第一问的追问" if history and question == "它呢" else question
    ))
    seen = []

    def fake_answer(question, prompt_question=None):
        seen.append((question, prompt_question))
        yield {"type": "stage", "data": {"stage": "actor", "status": "running"}}
        yield {"type": "sources", "data": []}
        yield {"type": "content", "data": "完成"}

    monkeypatch.setattr(chat, "stream_answer", fake_answer)
    from app.main import app
    with TestClient(app) as client:
        first = client.post("/api/sessions").json()["id"]
        second = client.post("/api/sessions").json()["id"]
        response = client.post("/api/chat/stream", json={"session_id": first, "question": "第一问"})
        frames = [json.loads(part.removeprefix("data: ")) for part in response.text.strip().split("\n\n")]
        assert [f["type"] for f in frames][:3] == ["stage", "memory", "stage"]
        assert frames[-1]["type"] == "done"
        assert [r["content"] for r in session_service.get_session_messages(first)] == ["第一问", "完成"]
        assert session_service.get_session_messages(second) == []
        assert client.post("/api/chat/stream", json={"session_id": "unknown", "question": "x"}).status_code == 404
        client.post("/api/chat/stream", json={"session_id": first, "question": "它呢"})
        assert seen[-1][0] == "关于第一问的追问"
        assert "第一问" in seen[-1][1]
        assert "完成" in seen[-1][1]

        def failing_answer(*args, **kwargs):
            yield {"type": "content", "data": "partial"}
            raise RuntimeError("private upstream failure")

        monkeypatch.setattr(chat, "stream_answer", failing_answer)
        response = client.post("/api/chat/stream", json={"session_id": first, "question": "失败问题"})
        assert '"type": "error"' in response.text
        assert "private upstream failure" not in response.text
        assert len(session_service.get_session_messages(first)) == 4

        monkeypatch.setattr(chat, "stream_answer", lambda *a, **k: iter([{"type": "sources", "data": []}]))
        response = client.post("/api/chat/stream", json={"session_id": first, "question": "空回答"})
        assert '"type": "error"' in response.text
        assert '"type": "done"' not in response.text
        assert len(session_service.get_session_messages(first)) == 4


def test_recent_history_is_bounded_and_per_session(monkeypatch, tmp_path):
    monkeypatch.setattr(session_service, "DB_PATH", tmp_path / "history.db")
    session_service.init_db()
    first = session_service.create_session()["id"]
    second = session_service.create_session()["id"]
    session_service.add_exchange(first, "讲讲 Chroma", "Chroma 是向量库")
    session_service.add_exchange(second, "私有问题", "私有答案")
    assert [m["content"] for m in session_service.get_recent_turns(first)] == ["讲讲 Chroma", "Chroma 是向量库"]
    assert len(session_service.get_recent_turns(first, max_chars=6)[0]["content"]) <= 6
    assert "私有" not in str(session_service.get_recent_turns(first))


def test_referential_followup_uses_history_but_self_contained_question_does_not(monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Chroma 的索引如何持久化？",
        ))])

    monkeypatch.setattr(memory_service._client.chat.completions, "create", create)
    history = [{"role": "user", "content": "讲讲 Chroma"}]
    assert memory_service.resolve_question("它的索引如何持久化？", history) == "Chroma 的索引如何持久化？"
    assert len(requests) == 1
    assert memory_service.resolve_question("Neo4j 如何建图？", history) == "Neo4j 如何建图？"
    assert len(requests) == 1
    prompt = memory_service.generation_context("它的索引如何持久化？", "Chroma 的索引如何持久化？", history)
    assert "不能作为知识库事实证据" in prompt


def test_agent_rejects_invalid_tool_and_stops_repeat(monkeypatch, tmp_path):
    with pytest.raises(ToolError, match="缺少有效参数"):
        execute_tool("generate_report", {"title": "../x", "content": ""}, tmp_path)
    with pytest.raises(ToolError, match="未知工具"):
        execute_tool("arbitrary", {}, tmp_path)

    calls = []
    tc = SimpleNamespace(id="tool-1", function=SimpleNamespace(name="generate_report",
                         arguments=json.dumps({"title": "报告", "content": "正文"})))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, tool_calls=[tc],
    ))])
    end = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))])
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="回答"))])]

    def create(**kwargs):
        if kwargs.get("stream"):
            return chunks
        calls.append(kwargs)
        return response if len(calls) <= 2 else end

    monkeypatch.setattr(agent_service._llm_client.chat.completions, "create", create)
    events = list(agent_service.run_agent("写报告", tmp_path))
    results = [e["data"] for e in events if e["type"] == "tool_result"]
    assert [r["ok"] for r in results] == [True, False]
    assert len(list(Path(tmp_path).glob("*.md"))) == 1
    assert events[-1] == {"type": "content", "data": "回答"}


def test_answer_metrics_do_not_claim_keyword_overlap_is_correctness():
    gold = {"id": "1", "expected_route": "vector", "answerable": "1",
            "gold_keywords": "向量|检索", "gold_source": "doc.txt"}
    scored = score_case(gold, {"answer": "向量检索【来源：doc.txt 第1段】", "route": "vector",
                               "sources": [{"filename": "doc.txt"}], "latency_ms": 100})
    summary = summarize([scored])
    assert summary["route_accuracy"] == 1
    assert summary["keyword_recall_diagnostic"] == 1
    assert summary["judge_correctness"] is None


def test_combined_sources_require_both_exact_matches():
    gold = {"id": "h", "expected_route": "hybrid", "answerable": "1",
            "gold_source": "agent文档.docx + knowledge_graph"}
    prediction = {"answer": "有依据", "route": "hybrid",
                  "sources": [{"filename": "agent文档.docx"}]}
    assert score_case(gold, prediction)["source_match"] is False
    prediction["evidence"] = [{"filename": "知识图谱"}]
    assert score_case(gold, prediction)["source_match"] is True
    prediction["sources"] = [{"filename": "agent文档.doc"}]
    assert score_case(gold, prediction)["source_match"] is False


def test_score_mode_writes_reproducible_artifacts(monkeypatch, tmp_path):
    from evaluation import evaluate_answers
    dataset = tmp_path / "gold.csv"
    predictions = tmp_path / "predictions.csv"
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "expected_route", "gold_answer",
                                                    "answerable", "gold_source", "gold_keywords", "review_status"])
        writer.writeheader()
        writer.writerow({"id": "1", "question": "问", "expected_route": "vector",
                         "gold_answer": "答案", "answerable": "1", "gold_source": "doc.txt",
                         "gold_keywords": "答案", "review_status": "ready"})
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer", "route", "sources_json", "latency_ms"])
        writer.writeheader()
        writer.writerow({"id": "1", "answer": "答案【来源：doc.txt 第1段】", "route": "vector",
                         "sources_json": '[{"filename":"doc.txt"}]', "latency_ms": "20"})
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--mode", "score",
                                     "--dataset", str(dataset), "--predictions", str(predictions),
                                     "--output-dir", str(output)])
    evaluate_answers.main()
    summary = json.loads((output / "answer_summary.json").read_text())
    assert summary["completed"] == 1
    assert summary["judge_count"] == 0
    assert (output / "answer_case_scores.csv").exists()
