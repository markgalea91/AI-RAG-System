# src/rag_platform/evaluation/ragas_service.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_platform.common import utils

# keep your EmbeddingsAdapter exactly (move it to common if you want)
import asyncio
from typing import List as TList

class EmbeddingsAdapter:
    def __init__(self, lc_embeddings):
        self._lc = lc_embeddings

    def embed_query(self, text: str): return self._lc.embed_query(text)
    def embed_documents(self, texts: TList[str]): return self._lc.embed_documents(texts)
    def embed_text(self, text: str): return self._lc.embed_query(text)
    def embed_texts(self, texts: TList[str]): return self._lc.embed_documents(texts)

    async def _to_thread(self, fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def aembed_query(self, text: str): return await self._to_thread(self._lc.embed_query, text)
    async def aembed_documents(self, texts: TList[str]): return await self._to_thread(self._lc.embed_documents, texts)
    async def aembed_text(self, text: str): return await self._to_thread(self._lc.embed_query, text)
    async def aembed_texts(self, texts: TList[str]): return await self._to_thread(self._lc.embed_documents, texts)


class RagasEvalService:
    def __init__(
        self,
        *,
        ollama_openai_base_url: str = "http://localhost:11434/v1",
        ollama_api_key: str = "ollama",
        judge_model: str = "qwen2.5:7b-instruct",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
        self.client = AsyncOpenAI(base_url=ollama_openai_base_url, api_key=ollama_api_key)
        self.judge_llm = llm_factory(model=judge_model, client=self.client)
        lc_emb = HuggingFaceEmbeddings(model_name=embed_model)
        self.emb = EmbeddingsAdapter(lc_emb)

    def evaluate_jsonl(self, *, saved_outputs_path: str, max_workers: int = 1, timeout: int = 300):
        raw_rows = utils.read_jsonl(saved_outputs_path)

        rows = []
        for r in raw_rows:
            gt = r.get("ground_truths", [])
            reference = gt[0] if isinstance(gt, list) and gt else ""
            rows.append({
                "user_input": r.get("question", ""),
                "response": r.get("answer", ""),
                "retrieved_contexts": r.get("contexts", []),
                "reference": reference,
            })

        dataset = Dataset.from_list(rows)

        # inject dependencies
        context_precision.llm = self.judge_llm
        context_recall.llm = self.judge_llm
        faithfulness.llm = self.judge_llm

        answer_relevancy.llm = self.judge_llm
        answer_relevancy.embeddings = self.emb

        answer_correctness.llm = self.judge_llm
        answer_correctness.embeddings = self.emb

        metrics = [context_precision, context_recall, answer_relevancy, answer_correctness]
        return evaluate(dataset=dataset, metrics=metrics, show_progress=True, run_config=RunConfig(max_workers=max_workers, timeout=timeout))