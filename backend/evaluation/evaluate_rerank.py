# -*- coding: utf-8 -*-
"""
Rerank Evaluation V1

Goal
----
Evaluate ONLY the reranking stage while keeping the Vector Retrieval V1
candidate set frozen.

Important experimental rule:
- Do NOT change Chroma.
- Do NOT change chunk_size / chunk_overlap.
- Do NOT change the 42-question benchmark.
- Do NOT run a new vector search for candidate generation.
- Read the exact Top-5 candidates saved by Vector Retrieval V1 and only
  reorder those candidates with a CrossEncoder reranker.

This makes the difference between Vector V1 and Rerank V1 attributable to
reranking rather than to retrieval changes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _as_int(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _as_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _load_full_chunk_map() -> dict[tuple[str, int], str]:
    """
    Load current Chroma documents only to recover the FULL text for the
    already-frozen candidate chunk indices.

    This function does NOT call vector search and does NOT change candidate
    membership/order.
    """
    try:
        from app.services.vector_store_service import get_collection

        collection = get_collection()
        data = collection.get(include=["documents", "metadatas"])

        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        chunk_map: dict[tuple[str, int], str] = {}

        for doc, meta in zip(docs, metas):
            meta = meta or {}
            filename = meta.get("filename")
            chunk_index = _as_int(meta.get("chunk_index"))

            if filename is None or chunk_index is None:
                continue

            chunk_map[(str(filename), chunk_index)] = str(doc or "")

        return chunk_map
    except Exception as exc:
        print(f"[WARN] Could not load full chunks from Chroma: {exc}")
        print("[WARN] Falling back to rankN_content_preview stored in predictions CSV.")
        return {}


def _extract_frozen_candidates(
    row: pd.Series,
    candidate_k: int,
    full_chunk_map: dict[tuple[str, int], str],
) -> list[dict]:
    hits = []

    for rank in range(1, candidate_k + 1):
        filename = row.get(f"rank{rank}_filename")
        chunk_index = _as_int(row.get(f"rank{rank}_chunk_index"))

        if chunk_index is None:
            continue

        filename = "" if pd.isna(filename) else str(filename)

        preview = row.get(f"rank{rank}_content_preview")
        preview = "" if pd.isna(preview) else str(preview)

        content = full_chunk_map.get((filename, chunk_index), preview)

        hits.append(
            {
                "baseline_rank": rank,
                "filename": filename,
                "chunk_index": chunk_index,
                "baseline_score": _as_float(row.get(f"rank{rank}_score")),
                "content": content,
            }
        )

    return hits


def _find_gold_rank(candidates: list[dict], gold_chunk_index: int) -> int | None:
    for rank, hit in enumerate(candidates, start=1):
        if hit["chunk_index"] == gold_chunk_index:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="evaluation/fixtures/vector_retrieval_v1_top5.csv",
        help="Frozen Vector Retrieval V1 predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/rerank/v1",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-reranker-base",
        help="CrossEncoder reranker model.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=5,
        help="Rerank only the frozen Top-K candidates. Keep 5 for the clean V1 experiment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit. Use 0 for the formal run.",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path, encoding="utf-8-sig")
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    required = {"id", "question", "gold_chunk_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.candidate_k != 5:
        print(
            "[WARN] candidate-k is not 5. For the clean comparison with Vector V1, "
            "use --candidate-k 5."
        )

    # Load exact chunk text by frozen candidate index; no vector search is run here.
    full_chunk_map = _load_full_chunk_map()

    print("=" * 72)
    print("Rerank Evaluation V1")
    print("=" * 72)
    print(f"Predictions : {predictions_path}")
    print(f"Questions   : {len(df)}")
    print(f"Candidate K : {args.candidate_k}")
    print(f"Model       : {args.model}")

    device = _resolve_device(args.device)
    print(f"Device      : {device}")
    print()
    print("Experimental control:")
    print("  - Vector candidate set is FROZEN from V1 predictions.")
    print("  - Chroma search is NOT rerun.")
    print("  - Only candidate ORDER is changed.")
    print()

    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError(
            "Could not import sentence_transformers.CrossEncoder. "
            "Install/repair sentence-transformers first."
        ) from exc

    # Model load is reported separately from per-query warm latency.
    t0 = time.perf_counter()
    reranker = CrossEncoder(
        args.model,
        max_length=512,
        device=device,
    )
    model_load_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup so query-level latency does not include model initialization.
    warmup_start = time.perf_counter()
    _ = reranker.predict(
        [["warmup query", "warmup document"]],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    rows_out = []
    rerank_latencies = []

    baseline_rrs = []
    rerank_rrs = []

    baseline_hit1 = []
    baseline_hit3 = []
    baseline_hit5 = []

    rerank_hit1 = []
    rerank_hit3 = []
    rerank_hit5 = []

    rescue_count = 0
    harm_count = 0
    unchanged_hit1_count = 0
    candidate_miss_count = 0

    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        qid = row.get("id")
        question = str(row.get("question", ""))
        gold_chunk_index = _as_int(row.get("gold_chunk_index"))

        if gold_chunk_index is None:
            raise ValueError(f"id={qid}: gold_chunk_index is missing")

        candidates = _extract_frozen_candidates(
            row,
            args.candidate_k,
            full_chunk_map,
        )

        if not candidates:
            raise ValueError(f"id={qid}: no frozen candidates found")

        baseline_rank = _find_gold_rank(candidates, gold_chunk_index)
        baseline_rr = 0.0 if baseline_rank is None else 1.0 / baseline_rank

        baseline_h1 = baseline_rank == 1
        baseline_h3 = baseline_rank is not None and baseline_rank <= 3
        baseline_h5 = baseline_rank is not None and baseline_rank <= 5

        if baseline_rank is None:
            candidate_miss_count += 1

        pairs = [[question, hit["content"]] for hit in candidates]

        t1 = time.perf_counter()
        scores = reranker.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        rerank_ms = (time.perf_counter() - t1) * 1000.0
        rerank_latencies.append(rerank_ms)

        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"id={qid}: reranker returned {len(scores)} scores "
                f"for {len(candidates)} candidates"
            )

        for hit, score in zip(candidates, scores):
            hit["rerank_score"] = float(score)

        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        rerank_rank = _find_gold_rank(reranked, gold_chunk_index)
        rerank_rr = 0.0 if rerank_rank is None else 1.0 / rerank_rank

        rerank_h1 = rerank_rank == 1
        rerank_h3 = rerank_rank is not None and rerank_rank <= 3
        rerank_h5 = rerank_rank is not None and rerank_rank <= 5

        baseline_rrs.append(baseline_rr)
        rerank_rrs.append(rerank_rr)

        baseline_hit1.append(baseline_h1)
        baseline_hit3.append(baseline_h3)
        baseline_hit5.append(baseline_h5)

        rerank_hit1.append(rerank_h1)
        rerank_hit3.append(rerank_h3)
        rerank_hit5.append(rerank_h5)

        if (not baseline_h1) and rerank_h1:
            rescue_count += 1
        elif baseline_h1 and (not rerank_h1):
            harm_count += 1
        elif baseline_h1 and rerank_h1:
            unchanged_hit1_count += 1

        out = {
            "id": qid,
            "question": question,
            "gold_chunk_index": gold_chunk_index,
            "candidate_k": len(candidates),
            "baseline_gold_rank": baseline_rank,
            "rerank_gold_rank": rerank_rank,
            "baseline_rr": baseline_rr,
            "rerank_rr": rerank_rr,
            "baseline_hit@1": baseline_h1,
            "rerank_hit@1": rerank_h1,
            "baseline_hit@3": baseline_h3,
            "rerank_hit@3": rerank_h3,
            "baseline_hit@5": baseline_h5,
            "rerank_hit@5": rerank_h5,
            "rerank_latency_ms": rerank_ms,
            "rank_change": (
                None
                if baseline_rank is None or rerank_rank is None
                else baseline_rank - rerank_rank
            ),
        }

        for rank in range(1, min(args.candidate_k, 5) + 1):
            if rank <= len(reranked):
                hit = reranked[rank - 1]
                out[f"rerank_rank{rank}_chunk_index"] = hit["chunk_index"]
                out[f"rerank_rank{rank}_score"] = hit["rerank_score"]
                out[f"rerank_rank{rank}_baseline_rank"] = hit["baseline_rank"]
                out[f"rerank_rank{rank}_content_preview"] = hit["content"][:500]

        rows_out.append(out)

        print(
            f"[{pos:02d}/{len(df):02d}] id={qid} "
            f"baseline_rank={baseline_rank} rerank_rank={rerank_rank} "
            f"rerank_latency={rerank_ms:.1f} ms"
        )

    n = len(df)

    baseline_h1_rate = sum(baseline_hit1) / n
    baseline_h3_rate = sum(baseline_hit3) / n
    baseline_h5_rate = sum(baseline_hit5) / n
    baseline_mrr = sum(baseline_rrs) / n

    rerank_h1_rate = sum(rerank_hit1) / n
    rerank_h3_rate = sum(rerank_hit3) / n
    rerank_h5_rate = sum(rerank_hit5) / n
    rerank_mrr = sum(rerank_rrs) / n

    summary = {
        "questions": n,
        "model": args.model,
        "device": device,
        "candidate_k": args.candidate_k,
        "model_load_ms": model_load_ms,
        "warmup_ms": warmup_ms,
        "rerank_latency_p50_ms": _percentile(rerank_latencies, 50),
        "rerank_latency_p95_ms": _percentile(rerank_latencies, 95),
        "baseline": {
            "hit@1": baseline_h1_rate,
            "hit@3": baseline_h3_rate,
            "hit@5": baseline_h5_rate,
            "mrr": baseline_mrr,
        },
        "rerank": {
            "hit@1": rerank_h1_rate,
            "hit@3": rerank_h3_rate,
            "hit@5": rerank_h5_rate,
            "mrr": rerank_mrr,
        },
        "delta": {
            "hit@1_pp": (rerank_h1_rate - baseline_h1_rate) * 100.0,
            "hit@3_pp": (rerank_h3_rate - baseline_h3_rate) * 100.0,
            "hit@5_pp": (rerank_h5_rate - baseline_h5_rate) * 100.0,
            "mrr": rerank_mrr - baseline_mrr,
        },
        "rescue_count": rescue_count,
        "harm_count": harm_count,
        "candidate_miss_count": candidate_miss_count,
        "baseline_hit1_preserved_count": unchanged_hit1_count,
    }

    # Sanity invariant: reranking a fixed Top-5 candidate set cannot change Hit@5.
    invariant_ok = abs(rerank_h5_rate - baseline_h5_rate) < 1e-12
    summary["fixed_candidate_hit5_invariant_ok"] = invariant_ok

    predictions_out = output_dir / "rerank_predictions.csv"
    metrics_out = output_dir / "rerank_metrics.csv"
    summary_out = output_dir / "rerank_summary.json"

    pd.DataFrame(rows_out).to_csv(
        predictions_out,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df = pd.DataFrame(
        [
            {
                "metric": "Hit@1",
                "baseline": baseline_h1_rate,
                "rerank": rerank_h1_rate,
                "delta": rerank_h1_rate - baseline_h1_rate,
            },
            {
                "metric": "Hit@3",
                "baseline": baseline_h3_rate,
                "rerank": rerank_h3_rate,
                "delta": rerank_h3_rate - baseline_h3_rate,
            },
            {
                "metric": "Hit@5",
                "baseline": baseline_h5_rate,
                "rerank": rerank_h5_rate,
                "delta": rerank_h5_rate - baseline_h5_rate,
            },
            {
                "metric": "MRR",
                "baseline": baseline_mrr,
                "rerank": rerank_mrr,
                "delta": rerank_mrr - baseline_mrr,
            },
        ]
    )
    metrics_df.to_csv(metrics_out, index=False, encoding="utf-8-sig")

    summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Rerank Evaluation Summary")
    print("=" * 72)
    print(f"Questions                : {n}")
    print(f"Model                    : {args.model}")
    print(f"Device                   : {device}")
    print(f"Candidate set            : Frozen Vector V1 Top-{args.candidate_k}")
    print()
    print("Baseline Vector V1:")
    print(f"  Hit@1                  : {_percent(baseline_h1_rate)}")
    print(f"  Hit@3                  : {_percent(baseline_h3_rate)}")
    print(f"  Hit@5                  : {_percent(baseline_h5_rate)}")
    print(f"  MRR                    : {baseline_mrr:.4f}")
    print()
    print("After Rerank:")
    print(f"  Hit@1                  : {_percent(rerank_h1_rate)}")
    print(f"  Hit@3                  : {_percent(rerank_h3_rate)}")
    print(f"  Hit@5                  : {_percent(rerank_h5_rate)}")
    print(f"  MRR                    : {rerank_mrr:.4f}")
    print()
    print("Gain:")
    print(f"  Hit@1 delta            : {summary['delta']['hit@1_pp']:+.2f} pp")
    print(f"  Hit@3 delta            : {summary['delta']['hit@3_pp']:+.2f} pp")
    print(f"  Hit@5 delta            : {summary['delta']['hit@5_pp']:+.2f} pp")
    print(f"  MRR delta              : {summary['delta']['mrr']:+.4f}")
    print(f"  Rescue count           : {rescue_count}")
    print(f"  Harm count             : {harm_count}")
    print(f"  Candidate misses       : {candidate_miss_count}")
    print()
    print("Latency:")
    print(f"  Model load             : {model_load_ms:.1f} ms")
    print(f"  Warmup                 : {warmup_ms:.1f} ms")
    print(f"  Rerank P50             : {summary['rerank_latency_p50_ms']:.3f} ms")
    print(f"  Rerank P95             : {summary['rerank_latency_p95_ms']:.3f} ms")
    print()
    print(f"Fixed Top-{args.candidate_k} Hit@5 invariant: {'PASS' if invariant_ok else 'FAIL'}")
    print("=" * 72)
    print()
    print("Saved:")
    print(f"  {predictions_out}")
    print(f"  {metrics_out}")
    print(f"  {summary_out}")


if __name__ == "__main__":
    main()
