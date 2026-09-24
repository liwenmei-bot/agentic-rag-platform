"""Answer-level evaluation. Run from backend/; never calls the LLM in score mode.

Live: python -m evaluation.evaluate_answers --mode live --limit 20
Score: python -m evaluation.evaluate_answers --mode score --predictions evaluation/results/answers/answers.csv
Optional independent LLM judgement: add --judge to live mode (costs API calls).
"""
import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path

REFUSAL = re.compile(r"无法回答|无法确定|无法提供|没有检索到|未找到相关|资料不足|没有足够")
CITATION = re.compile(r"【来源：([^】]+)】")


def _json_field(value, default):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def score_case(gold: dict, prediction: dict) -> dict:
    answer = prediction.get("answer") or ""
    sources = _json_field(prediction.get("sources_json", prediction.get("sources")), [])
    evidence = _json_field(prediction.get("evidence_json", prediction.get("evidence")), [])
    keywords = [word.strip() for word in (gold.get("gold_keywords") or "").split("|") if word.strip()]
    expected_source = (gold.get("gold_source") or "").strip()
    answerable = str(gold.get("answerable", "")).strip() == "1"
    knowledge_required = str(gold.get("knowledge_required", "")).strip() == "1" or (
        not gold.get("knowledge_required") and gold.get("expected_route") in {"vector", "graph", "hybrid"}
    )
    refused = bool(REFUSAL.search(answer))
    cited = bool(CITATION.search(answer))
    source_names = [str(source.get("filename", "")) for source in sources]
    source_names += [str(item.get("filename", "")) for item in evidence]
    expected_sources = [part.strip() for part in expected_source.split("+") if part.strip()]
    actual_sources = set(source_names)
    source_match = (
        all(source in actual_sources or (source == "knowledge_graph" and "知识图谱" in actual_sources)
            for source in expected_sources)
        if expected_sources else None
    )
    return {
        "id": gold["id"],
        "route_match": str(prediction.get("route", "")).lower() == str(gold.get("expected_route", "")).lower(),
        "keyword_recall": (sum(word.casefold() in answer.casefold() for word in keywords) / len(keywords)) if keywords else None,
        "source_match": source_match,
        "citation_present": cited,
        "knowledge_required": knowledge_required,
        "refused": refused,
        "refusal_expected": not answerable,
        "refusal_correct": refused if not answerable else not refused,
        "latency_ms": float(prediction.get("latency_ms") or 0),
        "error": prediction.get("error", ""),
        "judge": _json_field(prediction.get("judge"), {}),
    }


def _mean(rows: list[dict], field: str):
    values = [r[field] for r in rows if r[field] is not None and not r["error"]]
    return round(statistics.mean(values), 4) if values else None


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row["error"]]
    latency = sorted(row["latency_ms"] for row in valid if row["latency_ms"] > 0)
    judged = [r["judge"] for r in valid if isinstance(r["judge"], dict) and "correct" in r["judge"]]
    faithful = [j["faithful"] for j in judged if isinstance(j.get("faithful"), bool)]
    knowledge = [row for row in rows if row["knowledge_required"]]
    direct = [row for row in rows if not row["knowledge_required"]]
    unanswerable = [row for row in rows if row["refusal_expected"]]
    answerable = [row for row in rows if not row["refusal_expected"]]
    return {
        "total": len(rows),
        "completed": len(valid),
        "failed": len(rows) - len(valid),
        "route_accuracy": _mean(rows, "route_match"),
        "keyword_recall_diagnostic": _mean(rows, "keyword_recall"),
        "source_match_rate": _mean(rows, "source_match"),
        "citation_rate_on_knowledge": _mean(knowledge, "citation_present"),
        "false_citation_rate_on_direct": _mean(direct, "citation_present"),
        "unanswerable_refusal_rate": _mean(unanswerable, "refused"),
        "answerable_overrefusal_rate": _mean(answerable, "refused"),
        "latency_p50_ms": round(statistics.median(latency), 2) if latency else None,
        "latency_p95_ms": round(latency[math.ceil(len(latency) * .95) - 1], 2) if latency else None,
        "judge_count": len(judged),
        "judge_correctness": round(statistics.mean(j["correct"] for j in judged), 4) if judged else None,
        "judge_faithfulness": round(statistics.mean(faithful), 4) if faithful else None,
        "note": "Keyword coverage is a diagnostic, not answer correctness. Judge scores require human spot checks.",
    }


def judge_answer(gold: dict, prediction: dict) -> dict:
    from openai import OpenAI
    from app.core.config import settings

    evidence = _json_field(prediction.get("evidence_json"), [])
    prompt = {
        "question": gold["question"],
        "reference_answer": gold.get("gold_answer"),
        "answerable": gold.get("answerable"),
        "model_answer": prediction.get("answer"),
        "evidence": evidence[:8],
    }
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=30, max_retries=1)
    response = client.chat.completions.create(
        model=settings.llm_model_name, temperature=0,
        messages=[
            {"role": "system", "content": (
                "评估回答是否满足参考答案（correct），以及是否被提供的实际检索证据支持（faithful）。"
                "证据为空且问题需要知识库事实时 faithful=false。"
                "仅输出 JSON：{\"correct\":true/false,\"faithful\":true/false,\"reason\":\"简短说明\"}。"
            )},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    result = json.loads(match.group() if match else raw)
    if not isinstance(result.get("correct"), bool) or not isinstance(result.get("faithful"), bool):
        raise ValueError("Judge returned invalid booleans")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "score"], required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("benchmark.csv"))
    parser.add_argument("--predictions", type=Path, help="Required in score mode; CSV with id/answer/route/sources_json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "answers")
    parser.add_argument("--limit", type=int, default=0, help="First N ready cases; 0 means all")
    parser.add_argument("--judge", action="store_true", help="Live mode only; calls the configured LLM")
    args = parser.parse_args()
    if args.mode == "score" and not args.predictions:
        parser.error("--predictions is required in score mode")
    if args.mode != "live" and args.judge:
        parser.error("--judge requires live mode")

    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        gold = [row for row in csv.DictReader(handle)
                if row.get("review_status", "ready") == "ready" and row.get("gold_answer")]
    if args.limit > 0:
        gold = gold[:args.limit]
    predictions = {}
    if args.mode == "score":
        with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
            predictions = {row["id"]: row for row in csv.DictReader(handle)}

    if args.mode == "live":
        from app.services.rag_service import answer_question
        for row in gold:
            started = time.perf_counter()
            try:
                result = answer_question(row["question"])
                prediction = {
                    "id": row["id"], "answer": result["answer"], "route": result["route"],
                    "sources_json": json.dumps(result.get("sources", []), ensure_ascii=False),
                    "evidence_json": json.dumps(result.get("evidence", []), ensure_ascii=False),
                    "error": "",
                }
                if args.judge:
                    try:
                        prediction["judge"] = json.dumps(judge_answer(row, prediction), ensure_ascii=False)
                    except Exception as exc:
                        prediction["judge"] = json.dumps({"error": type(exc).__name__})
            except Exception as exc:
                prediction = {"id": row["id"], "answer": "", "route": "", "error": type(exc).__name__}
            prediction["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            predictions[row["id"]] = prediction

    rows = [score_case(row, predictions.get(row["id"], {"error": "missing_prediction"})) for row in gold]
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "answer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "answer_case_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows({**row, "judge": json.dumps(row["judge"], ensure_ascii=False)} for row in rows)
    if args.mode == "live":
        with (args.output_dir / "answers.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "id", "answer", "route", "sources_json", "evidence_json", "latency_ms", "error", "judge",
            ])
            writer.writeheader()
            writer.writerows(predictions[row["id"]] for row in gold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
