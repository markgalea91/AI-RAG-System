# # src/rag_platform/retrieval/service.py
# from __future__ import annotations
#
# from typing import Optional, Tuple
# from langchain_ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from rag_platform.config.settings import get_settings
#
# from rag_platform.retrieval.milvus_client import MilvusRetriever
# from rag_platform.retrieval.prompting import PROMPT, hits_to_chunks, stuff_context, TRANSLATE_TO_EN_PROMPT, \
#     TRANSLATE_TO_MT_PROMPT, DETECT_LANG_PROMPT
# from rag_platform.retrieval.schemas import RetrievalResult, RetrievedChunk, RAGAnswer
# from langchain_core.messages import AIMessage
#
#
# def is_maltese_heuristic(text: str) -> bool:
#     return any(ch in text for ch in ("għ", "ċ", "ġ", "ħ", "ż", "Ċ", "Ġ", "Ħ", "Ż", "GĦ"))
#
#
# class RetrievalService:
#     def __init__(
#         self,
#         *,
#         retriever: MilvusRetriever,
#         llm_model: str,
#         translate_model: str,
#         temperature: float = 0.0,
#         reasoning: bool = False,
#     ):
#         settings = get_settings()
#
#         self.retriever = retriever
#         self.llm = ChatOllama(model=llm_model, temperature=temperature, reasoning=reasoning)
#         self.translator = ChatOllama(model=translate_model, temperature=temperature)
#
#         self.detect_lang = DETECT_LANG_PROMPT | self.translator | StrOutputParser()
#         self.to_en = TRANSLATE_TO_EN_PROMPT | self.translator | StrOutputParser()
#         self.to_mt = TRANSLATE_TO_MT_PROMPT | self.translator | StrOutputParser()
#
#         if reasoning:
#             self.chain = PROMPT | self.llm
#         else:
#             self.chain = PROMPT | self.llm | StrOutputParser()
#
#     def _detect_language(self, query: str) -> str:
#         # quick shortcut first
#         if is_maltese_heuristic(query):
#             return "mt"
#         # fallback to model detector
#         out = self.detect_lang.invoke({"text": query}).strip().lower()
#         return "mt" if out.startswith("mt") else "en"
#
#     def _query_for_retrieval(self, query: str) -> tuple[str, str]:
#         lang = self._detect_language(query)
#         if lang == "mt":
#             query_en = self.to_en.invoke({"text": query}).strip()
#             return query_en, "mt"
#         return query, "en"
#
#     def _translate_query_for_retrieval(self, query: str) -> Tuple[str, str]:
#         """
#         Returns (query_for_retrieval_en, user_lang).
#         user_lang is "mt" or "en" (basic).
#         """
#         if self._looks_maltese(query):
#             q_en = self.to_en.invoke({"text": query}).strip()
#             return q_en, "mt"
#         return query, "en"
#
#     def _translate_answer_back_if_needed(self, answer_text: str, user_lang: str) -> str:
#         """
#         Translate only the Answer: line back to Maltese.
#         Keep Sources (and Quote lines if present) as-is for audit.
#         """
#         if user_lang != "mt":
#             return answer_text
#
#         lines = answer_text.splitlines()
#         for i, line in enumerate(lines):
#             if line.strip().lower().startswith("answer:"):
#                 english_answer = line.split(":", 1)[1].strip()
#                 mt_answer = self.to_mt.invoke({"text": english_answer}).strip()
#                 lines[i] = f"Answer: {mt_answer}"
#                 break
#         return "\n".join(lines)
#
#     def answer(self, query: str, *, top_k: int = 3, max_chars_per_chunk: int = 1200) -> RAGAnswer:
#         hits = self.retriever.retrieve([query], top_k=top_k)[0]
#         chunk_dicts, sources = hits_to_chunks(hits, max_chars=max_chars_per_chunk)
#         stuffed = stuff_context(chunk_dicts)
#
#         if not chunk_dicts:
#             # follow your rule
#             answer_text = "I cannot answer this from the provided context."
#             reasoning_text = "None"
#         else:
#             msg = self.chain.invoke({"question": query, "context": stuffed})
#             answer_text = msg.content
#             reasoning_text = (msg.additional_kwargs.get("reasoning_content") or "").strip()
#
#         retrieval = RetrievalResult(
#             query=query,
#             stuffed_context=stuffed,
#             chunks=[RetrievedChunk(**c) for c in chunk_dicts],
#         )
#
#         return RAGAnswer(query=query, answer=answer_text, reasoning=reasoning_text, sources=sources, retrieval=retrieval)

# src/rag_platform/retrieval/service.py
from __future__ import annotations

from typing import Optional, Tuple, Union
import re

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from rag_platform.retrieval.milvus_client import MilvusRetriever
from rag_platform.retrieval.prompting import (
    PROMPT,
    hits_to_chunks,
    stuff_context,
    GENERIC_TRANSLATE_PROMPT,
    DETECT_LANG_PROMPT, ENGLISH_LANGUAGE, MALTESE_LANGUAGE, ENGLISH_CODE, MALTESE_CODE,
)
from rag_platform.retrieval.schemas import RetrievalResult, RetrievedChunk, RAGAnswer


def is_maltese_heuristic(text: str) -> bool:
    # Maltese-only letters: għ ċ ġ ħ ż (and uppercase variants)
    return any(ch in text for ch in ("għ", "ċ", "ġ", "ħ", "ż", "Ċ", "Ġ", "Ħ", "Ż", "GĦ"))


class RetrievalService:
    def __init__(
        self,
        *,
        retriever: MilvusRetriever,
        llm_model: str,
        translate_model: str,
        temperature: float = 0.0,
        reasoning: bool = False
    ):
        self.retriever = retriever
        self.reasoning_enabled = reasoning

        # Main RAG model
        self.llm = ChatOllama(model=llm_model, temperature=temperature, reasoning=reasoning)

        # Translator / language detector model (TranslateGemma etc.)
        # Keep temperature at 0 for deterministic translations/classification.
        self.translator = ChatOllama(model=translate_model, temperature=0.0)

        # Translation + language detection chains
        self.detect_lang = DETECT_LANG_PROMPT | self.translator | StrOutputParser()
        self.translate_text = GENERIC_TRANSLATE_PROMPT | self.translator | StrOutputParser()

        # RAG chain: keep AIMessage if reasoning=True so we can read thinking field.
        if reasoning:
            self.chain = PROMPT | self.llm
        else:
            self.chain = PROMPT | self.llm | StrOutputParser()

    # ---------- Language + translation helpers ----------

    def _detect_language(self, query: str) -> str:
        """
        Returns "mt" or "en".
        First do a cheap heuristic, then fallback to model-based detection.
        """
        if is_maltese_heuristic(query):
            return "mt"

        out = self.detect_lang.invoke({"text": query}).strip().lower()
        return "mt" if out.startswith("mt") else "en"

    def _query_for_retrieval(self, query: str) -> Tuple[str, str]:
        """
        Returns (query_for_retrieval_in_english, user_lang).
        If user_lang == "mt", we translate query -> English for retrieval.
        """
        user_lang = self._detect_language(query)
        if user_lang == "mt":
            query_en = self.translate_text.invoke(
                {"TEXT": query,
                 "SOURCE_LANG": MALTESE_LANGUAGE,
                 "SOURCE_CODE": MALTESE_CODE,
                 "TARGET_LANG": ENGLISH_LANGUAGE,
                 "TARGET_CODE": ENGLISH_CODE}
            ).strip()

            return query_en, "mt"
        return query, "en"

    def _translate_answer_line_to_maltese(self, answer_text: str) -> str:
        # """
        # Translate only the 'Answer:' line back to Maltese.
        # Keeps Sources (and any quotes) in English for auditability.
        # """
        # lines = answer_text.splitlines()
        # for i, line in enumerate(lines):
        #     if line.strip().lower().startswith("answer:"):
        #         english_answer = line.split(":", 1)[1].strip()
        #         mt_answer = self.translate_text.invoke(
        #             {"TEXT": english_answer,
        #              "SOURCE_LANG": ENGLISH_LANGUAGE,
        #              "SOURCE_CODE": ENGLISH_CODE,
        #              "TARGET_LANG": MALTESE_LANGUAGE,
        #              "TARGET_CODE": MALTESE_CODE}
        #         ).strip()
        #
        #         lines[i] = f"Answer: {mt_answer}"
        #         break
        # return "\n".join(lines)

        """
           Translate everything in the Answer section (after 'Answer:' up to 'Sources:' or end),
           keeping the 'Sources:' line untouched.
           """
        # Split into Answer part and Sources part (if present)
        m = re.search(r"(?is)\bSources:\b", answer_text)
        if m:
            answer_part = answer_text[: m.start()].rstrip()
            sources_part = answer_text[m.start():].lstrip()
        else:
            answer_part = answer_text.rstrip()
            sources_part = ""

        # Remove the leading "Answer:" label if present, translate the body, then restore label.
        if answer_part.strip().lower().startswith("answer:"):
            body = answer_part.split(":", 1)[1].strip()
            mt_body = self.translate_text.invoke(
                    {"TEXT": body,
                     "SOURCE_LANG": ENGLISH_LANGUAGE,
                     "SOURCE_CODE": ENGLISH_CODE,
                     "TARGET_LANG": MALTESE_LANGUAGE,
                     "TARGET_CODE": MALTESE_CODE}
                ).strip()
            mt_answer_part = f"Answer: {mt_body}"
        else:
            # fallback: translate entire answer_part
            mt_answer_part = self.translate_text.invoke({"text": answer_part}).strip()

        return (mt_answer_part + ("\n\n" + sources_part if sources_part else "")).strip()

    # ---------- Main entrypoint ----------

    def answer(self, query: str, *, top_k: int = 3, max_chars_per_chunk: int = 1200) -> RAGAnswer:
        # Translate query to English BEFORE retrieval if Maltese
        query_en, user_lang = self._query_for_retrieval(query)

        # Retrieve using English query (your corpus is English-only)
        hits = self.retriever.retrieve([query_en], top_k=top_k)[0]
        chunk_dicts, sources = hits_to_chunks(hits, max_chars=max_chars_per_chunk)
        stuffed = stuff_context(chunk_dicts)

        reasoning_text: Optional[str] = None

        if not chunk_dicts:
            answer_text = "I cannot answer this from the provided context."
        else:
            # Generate using English question (matches English context)
            result = self.chain.invoke({"question": query_en, "context": stuffed})

            if self.reasoning_enabled:
                # result is an AIMessage
                assert isinstance(result, AIMessage)
                answer_text = result.content
                reasoning_text = (result.additional_kwargs or {}).get("reasoning_content")
                if isinstance(reasoning_text, str):
                    reasoning_text = reasoning_text.strip() or None
                else:
                    reasoning_text = None
            else:
                # result is a string
                answer_text = str(result)

            # Translate back if query was in Maltese
            if user_lang == "mt":
                answer_text = self._translate_answer_line_to_maltese(answer_text)

        retrieval = RetrievalResult(
            query=query,  # keep original user query
            stuffed_context=stuffed,
            chunks=[RetrievedChunk(**c) for c in chunk_dicts],
        )

        # If your schema allows, it's useful to store query_en too (debugging).
        # If not, leave it out to avoid breaking your schema.
        return RAGAnswer(
            query=query,
            answer=answer_text,
            reasoning=reasoning_text,
            sources=sources,
            retrieval=retrieval,
        )