from __future__ import annotations

import uuid
from fastapi import Request, Response
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from rag_platform.config.settings import get_settings, Settings
from rag_platform.retrieval.milvus_client import MilvusRetriever
from rag_platform.retrieval.service import RetrievalService
from rag_platform.retrieval.schemas import RAGAnswer

router = APIRouter(prefix="/rag", tags=["rag"])

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    llm_model: str = Field(..., min_length=1)
    reasoning: bool
    provider: str = Field(..., min_length=1)
    dotnet_is_chat: bool
    dotnet_client_guid: str = Field(..., min_length=1)



def get_retriever(milvus_uri: str, collection_name:str, hybrid: bool) -> MilvusRetriever:
    s = get_settings()
    return MilvusRetriever(
        milvus_uri=milvus_uri,
        collection_name=collection_name,  # make sure settings has this
        hybrid=hybrid,
    )


def get_service(
    retriever: MilvusRetriever,
        llm_model: str,
        translate_model: str,
        reasoning: bool,
        use_dotnet_llm: bool,
        provider: str,
        dotnet_base_url: str,
        dotnet_is_chat: bool,
        dotnet_client_guid: str,
        trace_id: str
) -> RetrievalService:
    return RetrievalService(
        retriever=retriever,
        llm_model=llm_model,  # make sure settings has this
        translate_model=translate_model,
        reasoning=reasoning,
        use_dotnet_llm=use_dotnet_llm,
        provider=provider,
        dotnet_base_url=dotnet_base_url,
        dotnet_is_chat=dotnet_is_chat,
        dotnet_client_guid=dotnet_client_guid,
        trace_id=trace_id
    )

@router.post("/query", response_model=RAGAnswer)
def query_rag(payload: RAGQueryRequest, request: Request, response: Response) -> RAGAnswer:

    settings = get_settings()
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    response.headers["X-Trace-Id"] = trace_id

    milvus_retrieval = get_retriever(settings.milvus_uri,
                                     settings.collection_name,
                                     settings.hybrid_search)

    svc = get_service(retriever=milvus_retrieval,
                      translate_model=settings.translator_model,
                      llm_model=payload.llm_model,
                      reasoning=payload.reasoning,
                      use_dotnet_llm=settings.use_api,
                      dotnet_base_url=settings.base_url,
                      dotnet_client_guid=payload.dotnet_client_guid,
                      provider=payload.provider,
                      dotnet_is_chat = payload.dotnet_is_chat,
                      trace_id=trace_id)

    ans = svc.answer(
        payload.query,
        top_k=settings.top_k,
        max_chars_per_chunk=settings.max_chars_per_chunk,
    )

    return ans