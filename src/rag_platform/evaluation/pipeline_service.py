# src/rag_platform/evaluation/pipeline_service.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from tqdm import tqdm

from rag_platform.retrieval.service import RetrievalService
from rag_platform.common.utils import read_jsonl, append_jsonl # or keep local helpers


def hits_to_context_list_from_retrieval_answer(retrieval_answer) -> List[str]:
    # retrieval_answer.retrieval.chunks already normalized by your service
    contexts = []
    for c in retrieval_answer.retrieval.chunks:
        header = f"[source: {c.source_name}" + (f", page: {c.page_number}]" if c.page_number is not None else "]")
        contexts.append(f"{header}\n{c.text}")
    return contexts


class EvalPipelineService:
    def __init__(self, *, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

    def run_and_save(
        self,
        *,
        testset_path: str,
        output_path: str,
        top_k: int = 3,
        resume: bool = True,
    ) -> None:
        processed_questions = set()
        if resume and os.path.exists(output_path):
            processed = read_jsonl(output_path)
            processed_questions = {r["question"] for r in processed}

        testset = read_jsonl(testset_path)

        for row in tqdm(testset, desc="Retrieval + Generation"):
            question = row["user_input"]
            if question in processed_questions:
                continue

            ground_truth = row.get("reference", "")

            rag = self.retrieval_service.answer(question, top_k=top_k)
            contexts = hits_to_context_list_from_retrieval_answer(rag)

            output_row = {
                "question": question,
                "answer": rag.answer,
                "contexts": contexts,
                "ground_truths": [ground_truth],
            }
            append_jsonl(output_path, output_row)