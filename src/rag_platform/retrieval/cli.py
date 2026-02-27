# src/rag_platform/retrieval/cli.py
from __future__ import annotations

import argparse
import json

from rag_platform.common.health import aggregate, check_milvus_uri, check_nim_embedding, check_ollama
from rag_platform.config.settings import get_settings
from rag_platform.retrieval.milvus_client import MilvusRetriever
from rag_platform.retrieval.service import RetrievalService


def main() -> None:

    s = get_settings()
    readiness = aggregate([
        check_milvus_uri(s.milvus_uri),
        check_nim_embedding(s.embedding_endpoint),
        check_ollama(getattr(s, "ollama_base_url", "http://localhost:11434")),
    ])
    if not readiness["ok"]:
        print(json.dumps(readiness, indent=2))
        raise SystemExit(2)

    p = argparse.ArgumentParser()
    p.add_argument("--milvus-uri", default="http://localhost:19530")
    p.add_argument("--collection", default="MTCA_ENG_HYB")
    p.add_argument("--hybrid", action="store_true", default=False)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--llm-model", default="nemotron-3-nano:30b")
    p.add_argument("--translate-model", default="translategemma:12b")
    p.add_argument("--reasoning", default=True)

    #EN
    # p.add_argument("--query", default="What is my TAX Rate as a parent as of 2026 and what is the difference with previous years  including children ?")
    # p.add_argument("--query", default="Which form is required by the VAT department for submitting a periodic statement summarising activities?")

    #MT
    p.add_argument("--query", default="X’inhi t-taxxa li rrid inhallas jien bhala genitur ghas-sena 2026 u x'inhi id-differenza min snin ta' qabel?")
    args = p.parse_args()

    retriever = MilvusRetriever(milvus_uri=args.milvus_uri, collection_name=args.collection, hybrid=args.hybrid)
    svc = RetrievalService(retriever=retriever, llm_model=args.llm_model, translate_model=args.translate_model, temperature=0.0, reasoning=args.reasoning)

    out = svc.answer(args.query, top_k=args.top_k)
    print(json.dumps(out.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()