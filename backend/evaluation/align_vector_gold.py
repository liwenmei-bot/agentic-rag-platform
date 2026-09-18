# -*- coding: utf-8 -*-
"""
Generate candidate chunk alignments for benchmark gold labels.

IMPORTANT:
This script does NOT modify benchmark.csv and does NOT automatically declare a
candidate as the gold chunk. It only produces review candidates based on the
current Chroma chunk content and the benchmark gold_source/gold_keywords fields.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent if EVAL_DIR.name == "evaluation" else Path.cwd()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.vector_store_service import get_collection


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _expected_filenames(gold_source: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*\+\s*|[|;]", _clean_text(gold_source)) if p.strip()]
    return [p for p in parts if p.casefold() not in {"knowledge_graph", "graph", "neo4j"}]


def _keywords(value) -> list[str]:
    return [x.strip() for x in _clean_text(value).split("|") if x.strip()]


def _load_chunks() -> list[dict]:
    collection = get_collection()
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    rows = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        rows.append(
            {
                "chroma_id": cid,
                "filename": _clean_text(meta.get("filename")),
                "doc_id": _clean_text(meta.get("doc_id")),
                "chunk_index": meta.get("chunk_index"),
                "content": doc or "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create vector gold alignment candidates")
    parser.add_argument("--benchmark", default="evaluation/benchmark.csv")
    parser.add_argument(
        "--output",
        default="evaluation/results/vector_retrieval/vector_gold_alignment_candidates.csv",
    )
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    benchmark = Path(args.benchmark)
    output = Path(args.output)
    if not benchmark.is_absolute():
        benchmark = BACKEND_DIR / benchmark
    if not output.is_absolute():
        output = BACKEND_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(benchmark)
    required = {"id", "question", "expected_route", "answerable", "gold_source", "gold_keywords", "gold_chunk_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Benchmark missing columns: {missing}")

    chunks = _load_chunks()
    if not chunks:
        raise RuntimeError("Chroma collection is empty; ingest the evaluation document first.")

    mask = (
        df["expected_route"].astype(str).str.lower().isin(["vector", "hybrid"])
        & (pd.to_numeric(df["answerable"], errors="coerce").fillna(0).astype(int) == 1)
    )
    targets = df.loc[mask].copy()

    out_rows = []
    for _, row in targets.iterrows():
        expected_files = _expected_filenames(row.get("gold_source"))
        keys = _keywords(row.get("gold_keywords"))

        candidates = [
            c for c in chunks
            if not expected_files
            or c["filename"].casefold() in {x.casefold() for x in expected_files}
        ]

        scored = []
        for c in candidates:
            content_fold = c["content"].casefold()
            matched = [k for k in keys if k.casefold() in content_fold]
            missing_keys = [k for k in keys if k.casefold() not in content_fold]
            coverage = (len(matched) / len(keys)) if keys else 0.0
            scored.append((coverage, len(matched), c, matched, missing_keys))

        scored.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                x[2].get("chunk_index") if isinstance(x[2].get("chunk_index"), int) else 10**12,
            )
        )

        if not scored:
            out_rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected_route": row["expected_route"],
                    "gold_source": row.get("gold_source"),
                    "gold_keywords": row.get("gold_keywords"),
                    "candidate_rank": 0,
                    "candidate_filename": "",
                    "candidate_doc_id": "",
                    "candidate_chunk_index": "",
                    "keyword_hits": 0,
                    "keyword_total": len(keys),
                    "keyword_coverage": 0.0,
                    "matched_keywords": "",
                    "missing_keywords": "|".join(keys),
                    "suggested_confidence": "none",
                    "content_preview": "NO CHUNKS FOUND FOR EXPECTED SOURCE",
                    "content": "",
                }
            )
            continue

        top = scored[: max(1, args.top_n)]
        second_coverage = top[1][0] if len(top) > 1 else -1.0
        for rank, (coverage, hit_count, c, matched, missing_keys) in enumerate(top, start=1):
            if rank == 1 and coverage == 1.0 and coverage > second_coverage:
                confidence = "high_unique_full_coverage"
            elif rank == 1 and coverage >= 0.5:
                confidence = "review_top_candidate"
            else:
                confidence = "review"

            out_rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected_route": row["expected_route"],
                    "gold_source": row.get("gold_source"),
                    "gold_keywords": row.get("gold_keywords"),
                    "candidate_rank": rank,
                    "candidate_filename": c["filename"],
                    "candidate_doc_id": c["doc_id"],
                    "candidate_chunk_index": c["chunk_index"],
                    "keyword_hits": hit_count,
                    "keyword_total": len(keys),
                    "keyword_coverage": round(coverage, 4),
                    "matched_keywords": "|".join(matched),
                    "missing_keywords": "|".join(missing_keys),
                    "suggested_confidence": confidence,
                    "content_preview": c["content"][:240].replace("\n", " "),
                    "content": c["content"],
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Benchmark : {benchmark}")
    print(f"Targets   : {len(targets)} answerable VECTOR/HYBRID questions")
    print(f"Candidates: {len(out_df)} rows")
    print(f"Saved     : {output}")
    print("\nNOTE: Review candidate_chunk_index manually before filling gold_chunk_index.")


if __name__ == "__main__":
    main()
