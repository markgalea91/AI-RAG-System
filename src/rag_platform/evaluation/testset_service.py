# src/rag_platform/evaluation/testset_service.py
from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.embeddings.base import embedding_factory
from ragas.run_config import RunConfig
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.extractors.llm_based import HeadlinesExtractor
from ragas.testset.transforms.splitters.headline import HeadlineSplitter
# from ragas.testset.synthesizers import (
#     SingleHopSpecificQuerySynthesizer,
#     MultiHopAbstractQuerySynthesizer
# )

from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer


class TestsetService:
    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_api_key: str = "ollama",
        ollama_model: str = "qwen2.5:7b-instruct",
        embedding_provider: str = "huggingface",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.ollama_model = ollama_model
        self.openai_client = OpenAI(base_url=ollama_base_url, api_key=ollama_api_key)
        self.llm = llm_factory(
            provider="openai",
            model=ollama_model,
            client=self.openai_client,
            adapter="instructor",
            temperature=0,
        )
        self.emb = embedding_factory(provider=embedding_provider, model=embedding_model)

    def relaxed_relationship_condition(rel) -> bool:
        # relationship properties may be stored in different places depending on version;
        # try both patterns.
        props = getattr(rel, "properties", None) or getattr(rel, "metadata", None) or {}
        cs = props.get("cosine_similarity", 0.0)
        ov = props.get("overlap_score", 0.0)
        return (cs >= 0.15) or (ov >= 0.05)

    def generate(
        self,
        *,
        documents,
        n: int,
        timeout: int = 180,
        max_retries: int = 5,
        max_wait: int = 10,
    ):
        run_config = RunConfig(timeout=timeout, max_retries=max_retries, max_wait=max_wait)
        generator = TestsetGenerator(llm=self.llm, embedding_model=self.emb)

        # build the defaults, then drop HeadlineSplitter
        transforms = default_transforms(
            documents=list(documents),
            llm=self.llm,
            embedding_model=self.emb,
        )
        # Drop headline-based steps
        transforms = [
            t for t in transforms
            if not isinstance(t, (HeadlinesExtractor, HeadlineSplitter))
        ]

        # qdist = default_query_distribution(llm=self.llm)
        # qdist = [
        #     (SingleHopSpecificQuerySynthesizer(llm=self.llm), 0.7),
        #     (MultiHopAbstractQuerySynthesizer(llm=self.llm), 0.3),
        #     # DO NOT include MultiHopSpecificQuerySynthesizer
        # ]

        qdist = [
            (SingleHopSpecificQuerySynthesizer(llm=self.llm), 0.8),
            (MultiHopSpecificQuerySynthesizer(llm=self.llm), 0.2),
        ]

        return generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=n,
            query_distribution=qdist,
            transforms=transforms,
            run_config=run_config
        )

    @staticmethod
    def write_jsonl_from_df(df, out_jsonl: str) -> None:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")