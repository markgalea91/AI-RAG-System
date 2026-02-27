# src/rag_platform/retrieval/milvus_client.py
from __future__ import annotations

from typing import Any, Dict, List
from nv_ingest_client.util.milvus import nvingest_retrieval


class MilvusRetriever:
    def __init__(self, *, milvus_uri: str, collection_name: str, hybrid: bool):
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.hybrid = hybrid

    def retrieve(self, queries: List[str], *, top_k: int) -> List[List[Dict[str, Any]]]:
        return nvingest_retrieval(
            queries,
            self.collection_name,
            milvus_uri=self.milvus_uri,
            hybrid=self.hybrid,
            top_k=top_k,
        )