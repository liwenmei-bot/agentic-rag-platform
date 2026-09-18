# -*- coding: utf-8 -*-
"""
Evaluate raw Chroma vector retrieval.

Strict Hit@K/MRR are computed ONLY for benchmark rows whose gold_chunk_index has
been aligned to the CURRENT Chroma collection. Until then, source-hit and
keyword-coverage metrics are diagnostics, not substitutes for strict chunk metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent if EVAL_DIR.name == "evaluation" else Path.cwd()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.vector_store_service import search


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _expected_filenames(value) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*\+\s*|[|;]", _clean(value)) if p.strip()]
    return [p for p in parts if p.casefold() not in {"knowledge_graph", "graph", "neo4j"}]


def _keywords(value) -> list[str]:
    return [x.strip() for x in _clean(value).split("|") if x.strip()]


def _gold_indices(value) -> set[int]:
    raw = _clean(value)
    if not raw:
        return set()
    values = re.split(r"[|,;\s]+", raw)
    out = set()
    for item in values:
        if not item:
            continue
        try:
            out.add(int(float(item)))
        except ValueError:
            pass
    return out


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - idx) + s[hi] * (idx - lo)


def _mean_bool(values: list[bool]) -> float | None:
    return (sum(bool(x) for x in values) / len(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector retrieval Hit@K and MRR")
    parser.add_argument("--benchmark", default="evaluation/benchmark.csv")
    parser.add_argument("--output-dir", default="evaluation/results/vector_retrieval/eval")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    benchmark = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    if not benchmark.is_absolute():
        benchmark = BACKEND_DIR / benchmark
    if not output_dir.is_absolute():
        output_dir = BACKEND_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(benchmark)
    required = {"id", "question", "expected_route", "answerable", "gold_source", "gold_keywords", "gold_chunk_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Benchmark missing columns: {missing}")

    # Vector branch is relevant to answerable VECTOR and HYBRID questions.
    answerable = pd.to_numeric(df["answerable"], errors="coerce").fillna(0).astype(int)
    route = df["expected_route"].astype(str).str.lower()
    eval_mask = answerable.eq(1) & route.isin(["vector", "hybrid"])
    eval_df = df.loc[eval_mask].copy()
    if args.limit > 0:
        eval_df = eval_df.head(args.limit)

    ks = [k for k in (1, 3, 5) if k <= args.max_k]
    if args.max_k not in ks:
        ks.append(args.max_k)
    ks = sorted(set(ks))

    rows = []
    errors = 0

    print(f"Benchmark : {benchmark}")
    print(f"Questions : {len(eval_df)} answerable VECTOR/HYBRID")
    print(f"Top-K     : {args.max_k}")
    print()

    for pos, (_, item) in enumerate(eval_df.iterrows(), start=1):
        q = _clean(item.get("question"))
        expected_files = _expected_filenames(item.get("gold_source"))
        expected_file_set = {x.casefold() for x in expected_files}
        keys = _keywords(item.get("gold_keywords"))
        gold_indices = _gold_indices(item.get("gold_chunk_index"))

        t0 = time.perf_counter()
        error = ""
        try:
            hits = search(q, top_k=args.max_k)
        except Exception as exc:
            hits = []
            error = f"{type(exc).__name__}: {exc}"
            errors += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        per_hit = []
        strict_rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            filename = _clean(hit.get("filename"))
            chunk_index = hit.get("chunk_index")
            try:
                chunk_index_int = int(chunk_index) if chunk_index is not None else None
            except (TypeError, ValueError):
                chunk_index_int = None

            content = _clean(hit.get("content"))
            source_match = (not expected_file_set) or filename.casefold() in expected_file_set
            matched_keys = [k for k in keys if k.casefold() in content.casefold()]
            keyword_coverage = (len(matched_keys) / len(keys)) if keys else 0.0
            strict_match = bool(
                gold_indices
                and source_match
                and chunk_index_int is not None
                and chunk_index_int in gold_indices
            )
            if strict_match and strict_rr == 0.0:
                strict_rr = 1.0 / rank

            per_hit.append(
                {
                    "rank": rank,
                    "filename": filename,
                    "doc_id": _clean(hit.get("doc_id")),
                    "chunk_index": chunk_index_int,
                    "score": hit.get("score"),
                    "source_match": source_match,
                    "strict_chunk_match": strict_match,
                    "keyword_coverage": keyword_coverage,
                    "matched_keywords": matched_keys,
                    "content": content,
                }
            )

        out = {
            "id": item.get("id"),
            "question": q,
            "expected_route": item.get("expected_route"),
            "gold_source": item.get("gold_source"),
            "gold_keywords": item.get("gold_keywords"),
            "gold_chunk_index": item.get("gold_chunk_index"),
            "strict_gold_available": bool(gold_indices),
            "latency_ms": round(latency_ms, 3),
            "returned_hits": len(per_hit),
            "error": error,
            "strict_rr": strict_rr if gold_indices else None,
        }

        for k in ks:
            top = per_hit[:k]
            out[f"source_hit@{k}"] = any(h["source_match"] for h in top)
            out[f"max_keyword_coverage@{k}"] = max(
                (h["keyword_coverage"] for h in top if h["source_match"]),
                default=0.0,
            )
            out[f"strict_hit@{k}"] = (
                any(h["strict_chunk_match"] for h in top) if gold_indices else None
            )

        for rank in range(1, args.max_k + 1):
            hit = per_hit[rank - 1] if len(per_hit) >= rank else None
            out[f"rank{rank}_filename"] = hit["filename"] if hit else ""
            out[f"rank{rank}_chunk_index"] = hit["chunk_index"] if hit else ""
            out[f"rank{rank}_score"] = hit["score"] if hit else ""
            out[f"rank{rank}_source_match"] = hit["source_match"] if hit else False
            out[f"rank{rank}_keyword_coverage"] = round(hit["keyword_coverage"], 4) if hit else 0.0
            out[f"rank{rank}_content_preview"] = hit["content"][:220].replace("\n", " ") if hit else ""

        rows.append(out)

        strict_label = "aligned" if gold_indices else "unaligned"
        print(
            f"[{pos:02d}/{len(eval_df):02d}] id={item.get('id')} "
            f"hits={len(per_hit)} {strict_label} latency={latency_ms:.1f} ms"
        )
        if args.delay > 0:
            time.sleep(args.delay)

    pred_df = pd.DataFrame(rows)
    pred_path = output_dir / "vector_retrieval_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = []
    strict_df = pred_df[pred_df["strict_gold_available"] == True].copy()  # noqa: E712

    for k in ks:
        metrics.append({"metric": f"source_hit@{k}", "value": pred_df[f"source_hit@{k}"].mean(), "n": len(pred_df), "status": "diagnostic"})
        metrics.append({"metric": f"avg_max_keyword_coverage@{k}", "value": pred_df[f"max_keyword_coverage@{k}"].mean(), "n": len(pred_df), "status": "diagnostic"})
        if len(strict_df):
            metrics.append({"metric": f"strict_hit@{k}", "value": strict_df[f"strict_hit@{k}"].astype(float).mean(), "n": len(strict_df), "status": "strict"})

    if len(strict_df):
        metrics.append({"metric": "strict_mrr", "value": strict_df["strict_rr"].astype(float).mean(), "n": len(strict_df), "status": "strict"})

    latencies = pred_df["latency_ms"].astype(float).tolist() if len(pred_df) else []
    metrics.extend(
        [
            {"metric": "latency_p50_ms", "value": _percentile(latencies, 0.50), "n": len(latencies), "status": "runtime"},
            {"metric": "latency_p95_ms", "value": _percentile(latencies, 0.95), "n": len(latencies), "status": "runtime"},
            {"metric": "errors", "value": errors, "n": len(pred_df), "status": "runtime"},
            {"metric": "strict_aligned_queries", "value": len(strict_df), "n": len(pred_df), "status": "coverage"},
        ]
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_dir / "vector_retrieval_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary = {
        "benchmark": str(benchmark),
        "questions": len(pred_df),
        "errors": errors,
        "strict_aligned_queries": len(strict_df),
        "strict_metrics_available": bool(len(strict_df)),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "metrics": {row["metric"]: row["value"] for row in metrics},
        "note": (
            "Strict Hit@K/MRR are valid only for rows with gold_chunk_index aligned "
            "to the current Chroma collection. Source-hit and keyword coverage are diagnostics."
        ),
    }
    summary_path = output_dir / "vector_retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("Vector Retrieval Evaluation Summary")
    print("=" * 68)
    print(f"Questions            : {len(pred_df)}")
    print(f"Errors               : {errors}")
    print(f"Strict aligned       : {len(strict_df)}/{len(pred_df)}")
    if len(strict_df):
        for k in ks:
            print(f"Strict Hit@{k:<2}         : {strict_df[f'strict_hit@{k}'].astype(float).mean():.2%}")
        print(f"Strict MRR           : {strict_df['strict_rr'].astype(float).mean():.4f}")
    else:
        print("Strict Hit@K / MRR   : NOT AVAILABLE (gold_chunk_index is not aligned yet)")
    for k in ks:
        print(f"Source Hit@{k:<2}         : {pred_df[f'source_hit@{k}'].mean():.2%}  [diagnostic]")
        print(f"Keyword coverage@{k:<2}   : {pred_df[f'max_keyword_coverage@{k}'].mean():.2%}  [diagnostic]")
    print(f"Latency P50          : {_percentile(latencies, 0.50):.3f} ms")
    print(f"Latency P95          : {_percentile(latencies, 0.95):.3f} ms")
    print("=" * 68)
    print("Saved:")
    print(f"  {pred_path}")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
