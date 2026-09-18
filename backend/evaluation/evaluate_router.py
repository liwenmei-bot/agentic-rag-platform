#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import rag_service  # noqa: E402


LABELS = ["direct", "vector", "graph", "hybrid"]
REQUIRED_COLUMNS = {"id", "question", "expected_route"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the real Agent Router in app.services.rag_service."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=EVAL_DIR / "benchmark.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR / "results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Sleep N seconds between questions.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs already present in router_predictions.csv.",
    )
    return parser.parse_args()


def load_benchmark(path: Path, limit: int | None):
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["expected_route"] = (
        df["expected_route"].astype(str).str.strip().str.lower()
    )

    invalid = sorted(set(df["expected_route"]) - set(LABELS))
    if invalid:
        raise ValueError(f"Invalid expected_route values: {invalid}")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in benchmark.csv")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        df = df.head(limit).copy()

    return df


def get_real_router():
    router = getattr(rag_service, "_route_query", None)
    if not callable(router):
        raise RuntimeError(
            "Cannot find app.services.rag_service._route_query(). "
            "Please confirm the router function name in rag_service.py."
        )
    return router


def evaluate_one(router, question: str):
    start = time.perf_counter()
    predicted_route = "error"
    route_reason = ""
    error = ""

    try:
        result = router(question)

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"_route_query() must return (route, reason), got: {result!r}"
            )

        predicted_route = str(result[0]).strip().lower()
        route_reason = str(result[1]).strip()

        if predicted_route not in LABELS:
            raise ValueError(f"Invalid route returned: {predicted_route!r}")

    except Exception as exc:
        predicted_route = "error"
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "predicted_route": predicted_route,
        "route_reason": route_reason,
        "latency_ms": round(latency_ms, 3),
        "error": error,
    }


def save_predictions(rows, path: Path):
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def safe_div(a, b):
    return float(a / b) if b else 0.0


def build_metrics(pred_df: pd.DataFrame):
    pred_df = pred_df.copy()

    pred_df["correct"] = (
        pred_df["expected_route"].astype(str).str.lower()
        == pred_df["predicted_route"].astype(str).str.lower()
    ).astype(int)

    metric_rows = []
    f1_values = []

    for label in LABELS:
        gold = pred_df["expected_route"] == label
        pred = pred_df["predicted_route"] == label

        tp = int((gold & pred).sum())
        fp = int((~gold & pred).sum())
        fn = int((gold & ~pred).sum())
        support = int(gold.sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        class_accuracy = safe_div(tp, support)

        f1_values.append(f1)

        metric_rows.append(
            {
                "scope": "route",
                "route": label,
                "support": support,
                "correct": tp,
                "accuracy": round(class_accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )

    overall_accuracy = float(pred_df["correct"].mean())
    macro_f1 = float(sum(f1_values) / len(f1_values))

    latency = pd.to_numeric(
        pred_df["latency_ms"], errors="coerce"
    ).dropna()

    summary = {
        "num_questions": int(len(pred_df)),
        "num_correct": int(pred_df["correct"].sum()),
        "num_errors": int((pred_df["predicted_route"] == "error").sum()),
        "router_accuracy": round(overall_accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "latency_ms": {
            "mean": round(float(latency.mean()), 3) if len(latency) else None,
            "p50": round(float(latency.quantile(0.50)), 3)
            if len(latency)
            else None,
            "p95": round(float(latency.quantile(0.95)), 3)
            if len(latency)
            else None,
        },
    }

    metric_rows.append(
        {
            "scope": "overall",
            "route": "ALL",
            "support": int(len(pred_df)),
            "correct": int(pred_df["correct"].sum()),
            "accuracy": round(overall_accuracy, 6),
            "precision": "",
            "recall": "",
            "f1": round(macro_f1, 6),
        }
    )

    return pred_df, pd.DataFrame(metric_rows), summary


def build_confusion_matrix(pred_df: pd.DataFrame):
    columns = LABELS.copy()
    if (pred_df["predicted_route"] == "error").any():
        columns.append("error")

    cm = pd.crosstab(
        pred_df["expected_route"],
        pred_df["predicted_route"],
        dropna=False,
    )

    cm = cm.reindex(index=LABELS, columns=columns, fill_value=0)
    cm.index.name = "expected_route"
    cm.columns.name = "predicted_route"

    return cm


def print_summary(summary, metrics_df):
    print()
    print("=" * 68)
    print("Agent Router Evaluation Summary")
    print("=" * 68)
    print(f"Questions       : {summary['num_questions']}")
    print(f"Correct         : {summary['num_correct']}")
    print(f"Errors          : {summary['num_errors']}")
    print(f"Router Accuracy : {summary['router_accuracy'] * 100:.2f}%")
    print(f"Macro-F1        : {summary['macro_f1'] * 100:.2f}%")

    latency = summary["latency_ms"]
    if latency["p50"] is not None:
        print(f"Latency P50     : {latency['p50']:.3f} ms")
        print(f"Latency P95     : {latency['p95']:.3f} ms")

    print()
    print("Per-route:")
    route_rows = metrics_df[metrics_df["scope"] == "route"]

    for _, row in route_rows.iterrows():
        print(
            f"  {row['route'].upper():<7} "
            f"support={int(row['support']):>2}  "
            f"acc={float(row['accuracy']) * 100:6.2f}%  "
            f"P={float(row['precision']) * 100:6.2f}%  "
            f"R={float(row['recall']) * 100:6.2f}%  "
            f"F1={float(row['f1']) * 100:6.2f}%"
        )

    print("=" * 68)


def main():
    args = parse_args()

    benchmark_path = args.benchmark.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "router_predictions.csv"
    metrics_path = output_dir / "router_metrics.csv"
    confusion_path = output_dir / "router_confusion_matrix.csv"
    summary_path = output_dir / "router_summary.json"

    benchmark_df = load_benchmark(benchmark_path, args.limit)
    router = get_real_router()

    print(f"Benchmark : {benchmark_path}")
    print(f"Rows      : {len(benchmark_df)}")
    print(f"Output    : {output_dir}")
    print("Router    : app.services.rag_service._route_query")
    print()

    completed = {}
    if args.resume and predictions_path.exists():
        old_df = pd.read_csv(predictions_path, encoding="utf-8-sig")
        if "id" in old_df.columns:
            for record in old_df.to_dict(orient="records"):
                completed[record["id"]] = record
        print(f"Resume: loaded {len(completed)} existing rows.")
        print()

    rows = []

    for pos, (_, bench_row) in enumerate(
        benchmark_df.iterrows(),
        start=1,
    ):
        case_id = bench_row["id"]

        if args.resume and case_id in completed:
            rows.append(completed[case_id])
            print(
                f"[{pos:02d}/{len(benchmark_df):02d}] "
                f"id={case_id} SKIP"
            )
            continue

        question = str(bench_row["question"])
        expected = str(bench_row["expected_route"]).lower()

        print(
            f"[{pos:02d}/{len(benchmark_df):02d}] "
            f"id={case_id}  {question}"
        )

        prediction = evaluate_one(router, question)
        predicted = prediction["predicted_route"]
        correct = int(predicted == expected)

        print(
            f"    expected={expected.upper():<6} "
            f"predicted={predicted.upper():<6} "
            f"correct={'YES' if correct else 'NO '} "
            f"latency={prediction['latency_ms']:.1f} ms"
        )

        if prediction["error"]:
            print(f"    ERROR: {prediction['error']}")
        elif prediction["route_reason"]:
            print(f"    reason: {prediction['route_reason']}")

        merged = bench_row.to_dict()
        merged.update(prediction)
        merged["correct"] = correct
        rows.append(merged)

        save_predictions(rows, predictions_path)

        if args.delay > 0 and pos < len(benchmark_df):
            time.sleep(args.delay)

    pred_df = save_predictions(rows, predictions_path)
    pred_df, metrics_df, summary = build_metrics(pred_df)
    confusion_df = build_confusion_matrix(pred_df)

    pred_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    confusion_df.to_csv(
        confusion_path,
        encoding="utf-8-sig",
    )

    settings = getattr(rag_service, "settings", None)
    summary["router_function"] = (
        "app.services.rag_service._route_query"
    )
    summary["llm_model_name"] = getattr(
        settings,
        "llm_model_name",
        None,
    )
    summary["benchmark_file"] = str(benchmark_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(summary, metrics_df)

    print()
    print("Saved:")
    print(f"  {predictions_path}")
    print(f"  {metrics_path}")
    print(f"  {confusion_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
