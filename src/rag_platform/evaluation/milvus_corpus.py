# src/rag_platform/evaluation/milvus_corpus.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from pymilvus import connections, Collection, utility

TEXT_CANDIDATE_FIELDS = ["text", "content", "page_content", "chunk", "chunk_text", "body"]


def _try_find_text(row: Dict[str, Any]) -> Optional[str]:
    for k in TEXT_CANDIDATE_FIELDS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    ent = row.get("entity")
    if isinstance(ent, dict):
        v = ent.get("text")
        if isinstance(v, str) and v.strip():
            return v.strip()

    best = None
    for k, v in row.items():
        if isinstance(v, str) and len(v.strip()) > 200:
            if best is None or len(v) > len(best):
                best = v.strip()
    return best


def _extract_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    md: Dict[str, Any] = {}

    for path in [
        ("source", "source_name"),
        ("source_metadata", "source_name"),
        ("metadata", "source_metadata", "source_name"),
        ("entity", "source", "source_name"),
    ]:
        cur: Any = row
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, str) and cur:
            md["source_name"] = cur
            break

    for path in [
        ("content_metadata", "page_number"),
        ("metadata", "content_metadata", "page_number"),
        ("entity", "content_metadata", "page_number"),
    ]:
        cur: Any = row
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, (int, float)):
            md["page_number"] = int(cur)
            break

    return md


def _milvus_iter_rows(
    collection: Collection,
    fields: List[str],
    batch_size: int = 512,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        remaining = None if max_rows is None else max_rows - len(out)
        if remaining is not None and remaining <= 0:
            break
        limit = batch_size if remaining is None else min(batch_size, remaining)

        rows = collection.query(expr="", output_fields=fields, limit=limit, offset=offset)
        if not rows:
            break
        out.extend(rows)
        offset += len(rows)

        if max_rows is not None and len(out) >= max_rows:
            break
    return out


def build_corpus_docs_from_milvus(
    *,
    milvus_uri: str,
    collection: str,
    min_chars: int = 200,
    page_size: int = 512,
    max_rows: int = 0,
    max_docs: int = 0,
    sample_docs: bool = False,
    seed: int = 42,
) -> List[Document]:
    random.seed(seed)

    uri = milvus_uri.replace("tcp://", "").replace("http://", "")
    host, port = uri.split(":")
    connections.connect(alias="default", host=host, port=port)

    if not utility.has_collection(collection):
        raise RuntimeError(f"Milvus collection not found: {collection}")

    col = Collection(collection)
    col.load()

    schema_fields = [f.name for f in col.schema.fields]
    fields = [f for f in schema_fields if "vector" not in f.lower() and "embedding" not in f.lower() and "sparse" not in f.lower()]

    max_rows_opt = None if max_rows <= 0 else max_rows
    rows = _milvus_iter_rows(col, fields=fields, batch_size=page_size, max_rows=max_rows_opt)

    docs: List[Document] = []
    seen: set[Tuple[str, str, Optional[int]]] = set()

    for r in rows:
        txt = _try_find_text(r)
        if not txt or len(txt) < min_chars:
            continue
        md = _extract_metadata(r)
        src = md.get("source_name", "unknown_source")
        page = md.get("page_number")
        key = (txt, src, page)
        if key in seen:
            continue
        seen.add(key)
        docs.append(Document(page_content=txt, metadata=md))

    if max_docs and max_docs > 0 and len(docs) > max_docs:
        docs = random.sample(docs, max_docs) if sample_docs else docs[:max_docs]

    if len(docs) < 30:
        raise RuntimeError("Too few text chunks found in Milvus to generate testset.")

    return docs