# src/rag_platform/evaluation/cli.py
from __future__ import annotations

import argparse

from pygments.lexer import default

from rag_platform.evaluation.milvus_corpus import build_corpus_docs_from_milvus
from rag_platform.evaluation.testset_service import TestsetService
from rag_platform.evaluation.pipeline_service import EvalPipelineService
from rag_platform.evaluation.ragas_service import RagasEvalService
from rag_platform.retrieval.milvus_client import MilvusRetriever
from rag_platform.retrieval.service import RetrievalService


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # --- testset ---
    t = sub.add_parser("testset")
    t.add_argument("--milvus-uri", default="tcp://localhost:19530")
    t.add_argument("--collection", default="MTCA_TEST_6")
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--min-chars", type=int, default=100)
    t.add_argument("--page-size", type=int, default=512)
    t.add_argument("--max-rows", type=int, default=0)
    t.add_argument("--max-docs", type=int, default=0)
    t.add_argument("--sample-docs", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    t.add_argument("--translate-model", default="translategemma:12b")
    t.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    t.add_argument("--out-csv", default="data/ragas_testset.csv")
    t.add_argument("--out-jsonl", default="data/ragas_testset.jsonl")

    # --- run pipeline ---
    r = sub.add_parser("run")
    r.add_argument("--testset", default="data/ragas_testset.jsonl")
    r.add_argument("--saved", default="data/pipeline_outputs.jsonl")
    r.add_argument("--milvus-uri", default="http://localhost:19530")
    r.add_argument("--collection", default="MTCA_TEST")
    r.add_argument("--hybrid", action="store_true")
    r.add_argument("--top-k", type=int, default=3)
    r.add_argument("--llm-model", default="nemotron-3-nano:30b")

    # --- eval ---
    e = sub.add_parser("eval")
    e.add_argument("--saved", default="data/pipeline_outputs.jsonl")
    e.add_argument("--out", default="result/ragas_results.csv")
    e.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    e.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = p.parse_args(["testset"])

    if args.cmd == "testset":
        docs = build_corpus_docs_from_milvus(
            milvus_uri=args.milvus_uri,
            collection=args.collection,
            min_chars=args.min_chars,
            page_size=args.page_size,
            max_rows=args.max_rows,
            max_docs=args.max_docs,
            sample_docs=args.sample_docs,
            seed=args.seed,
        )
        svc = TestsetService(ollama_model=args.ollama_model, embedding_model=args.embed_model)
        testset = svc.generate(documents=docs, n=args.n)
        df = testset.to_pandas()
        df.to_csv(args.out_csv, index=False)
        svc.write_jsonl_from_df(df, args.out_jsonl)
        print(f"[OK] wrote {args.out_csv} and {args.out_jsonl}")
        return

    if args.cmd == "run":
        retriever = MilvusRetriever(milvus_uri=args.milvus_uri, collection_name=args.collection, hybrid=args.hybrid)
        retrieval = RetrievalService(retriever=retriever, translate_model=args.translate_model, llm_model=args.llm_model, temperature=0.0)
        pipe = EvalPipelineService(retrieval_service=retrieval)
        pipe.run_and_save(testset_path=args.testset, output_path=args.saved, top_k=args.top_k, resume=True)
        print(f"[OK] wrote {args.saved}")
        return

    if args.cmd == "eval":
        svc = RagasEvalService(judge_model=args.judge_model, embed_model=args.embed_model)
        result = svc.evaluate_jsonl(saved_outputs_path=args.saved)
        df = result.to_pandas()
        df.to_csv(args.out, index=False)
        print(f"[OK] wrote {args.out}")
        print(result)
        return


if __name__ == "__main__":
    main()