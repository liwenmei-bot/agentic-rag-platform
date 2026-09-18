# -*- coding: utf-8 -*-
"""Export the current Chroma chunk inventory for retrieval evaluation."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent if EVAL_DIR.name == "evaluation" else Path.cwd()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.vector_store_service import get_collection


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect current Chroma chunks")
    parser.add_argument(
        "--output",
        default="evaluation/results/vector_retrieval/vector_chunk_catalog.csv",
        help="CSV output path",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = BACKEND_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    collection = get_collection()
    total = collection.count()

    print(f"Collection : {collection.name}")
    print(f"Chunks     : {total}")

    if total <= 0:
        print("Chroma collection is empty.")
        return

    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    rows = []
    for chunk_id, content, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        rows.append(
            {
                "chroma_id": chunk_id,
                "filename": meta.get("filename"),
                "doc_id": meta.get("doc_id"),
                "chunk_index": _safe_int(meta.get("chunk_index")),
                "content_chars": len(content or ""),
                "content": content or "",
            }
        )

    rows.sort(
        key=lambda r: (
            str(r.get("filename") or ""),
            r.get("chunk_index") if isinstance(r.get("chunk_index"), int) else 10**12,
            str(r.get("chroma_id") or ""),
        )
    )

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(r.get("filename") or "<missing filename>") for r in rows)
    print("\nBy filename:")
    for filename, count in sorted(counts.items()):
        print(f"  {filename}: {count}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
