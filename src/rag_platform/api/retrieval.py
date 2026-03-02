from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from rag_platform.config.settings import get_settings
from rag_platform.retrieval.milvus_client import MilvusRetriever
from rag_platform.retrieval.service import RetrievalService
from rag_platform.retrieval.schemas import RAGAnswer

router = APIRouter(prefix="/rag", tags=["rag"])


# Use your existing RetrievalQuery + add demo options
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=50)
    max_chars_per_chunk: int = Field(1200, ge=200, le=5000)
    include_raw: bool = False          # expose chunk.raw or not
    include_stuffed_context: bool = False  # show stuffed_context in debug panel


def get_retriever() -> MilvusRetriever:
    s = get_settings()
    return MilvusRetriever(
        milvus_uri=s.milvus_uri,
        collection_name=s.collection_name,  # make sure settings has this
        hybrid=getattr(s, "milvus_hybrid", False),
    )


def get_service(
    retriever: MilvusRetriever = Depends(get_retriever),
) -> RetrievalService:
    s = get_settings()
    return RetrievalService(
        retriever=retriever,
        llm_model=s.generation_model,  # make sure settings has this
        translate_model=s.translator_model,
        temperature=getattr(s, "temperature", 0.0),
        reasoning=s.reasoning,
        use_dotnet_llm=s.use_api,
        provider=s.provider,
        dotnet_base_url=s.base_url,
        dotnet_is_chat=s.is_chat,
        dotnet_client_guid=s.api_client_guid
    )


@router.post("/query", response_model=RAGAnswer)
def query_rag(
    payload: RAGQueryRequest,
    svc: RetrievalService = Depends(get_service),
) -> RAGAnswer:
    ans = svc.answer(
        payload.query,
        top_k=payload.top_k,
        max_chars_per_chunk=payload.max_chars_per_chunk,
    )

    # Optionally strip debug-heavy fields for demos
    if not payload.include_raw:
        for c in ans.retrieval.chunks:
            c.raw = None

    if not payload.include_stuffed_context:
        ans.retrieval.stuffed_context = ""

    return ans