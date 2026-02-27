# src/rag_platform/retrieval/prompting.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate


# PROMPT = ChatPromptTemplate.from_template(
#     """
#     /think
#
#     You are a helpful assistant.
#     Answer the question using ONLY the information provided in the context.
#
#     Rules:
#     - Every answer MUST contain an exact quote from the context.
#     - Below every answer there MUST be a sources/document names from where the answer was retrieved out of.
#     - Do NOT use prior knowledge or assumptions.
#     - If the question cannot be answered because the context discusses a different topic, reply exactly: "I cannot answer this from the provided context."
#
#     Question:
#     {question}
#
#     Answer
#
#     Context:
#     {context}
#
#     Evidence:
#     - [Source Name: ...] """
# )

# PROMPT = ChatPromptTemplate.from_template(
#     """
#     <|user|>
#     System: You are a helpful assistant.
#     Context: {context}
#     Question: {question}
#
#     Task: Answer the question using ONLY the information provided in the context.
#     Rules:
#      - Every answer MUST contain an exact quote from the context.
#      - Below every answer there MUST be a sources/document names from where the answer was retrieved out of.
#      - Do NOT use prior knowledge or assumptions.
#      - If the question cannot be answered because the context discusses a different topic, reply exactly: "I cannot answer this from the provided context."
#
#     <|assistant|>
#     <think>
#     Answer: [Final Answer]
#     Evidence: - [Source Name: ...]
#     """
# )

MALTESE_LANGUAGE = "Maltese"
MALTESE_CODE = "mt"

ENGLISH_LANGUAGE = "English"
ENGLISH_CODE = "en"

DETECT_LANG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You detect the language of the user's text. Reply with ONLY one token: mt or en. "
     "If mixed, reply with the dominant language."),
    ("user", "{text}")
])

GENERIC_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities. Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}"),
    ("user", "{TEXT}")
])

# TRANSLATE_TO_EN_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", "Translate the user's text to English. Output ONLY the translation. No explanations, no quotes, no extra text."),
#     ("user", "{text}")
# ])
#
# TRANSLATE_TO_MT_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", "Translate the user's text to Maltese. Output ONLY the translation. No explanations, no quotes, no extra text."),
#     ("user", "{text}")
# ])

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", """
Context:
{context}

Question:
{question}

Task:
Answer the question using ONLY the information provided in the context.

Rules:
- Identify the language of the question and answer in the same language.
- Below every answer there MUST be the sources/document names from where the answer was retrieved.
- Do NOT use prior knowledge or assumptions.
- If the question cannot be answered because the context discusses a different topic, reply exactly:
"I cannot answer this from the provided context."

Output format (exact):
Answer: ...

Sources: ...
""")
])

#


def hits_to_chunks(hits: List[Dict[str, Any]], *, max_chars: int = 1200) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      - list of normalized chunk dicts: {text, source_name, page_number, score, raw}
      - list of source_names (deduped, ordered)
    """
    chunks: List[Dict[str, Any]] = []
    sources: List[str] = []
    seen = set()

    for h in hits:
        ent = h.get("entity", {}) or {}
        text = (ent.get("text") or "").strip()
        if not text:
            continue

        src = ent.get("source", {}) or {}
        cm = ent.get("content_metadata", {}) or {}
        source_name = src.get("source_name", "unknown_source")
        page_number = cm.get("page_number", None)
        score = h.get("distance", None)

        if source_name not in seen:
            seen.add(source_name)
            sources.append(source_name)

        chunks.append(
            {
                "text": text[:max_chars],
                "source_name": source_name,
                "page_number": page_number,
                "score": score,
                "raw": h,
            }
        )

    return chunks, sources


def stuff_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for c in chunks:
        header = f"[source: {c['source_name']}" + (f", page: {c['page_number']}]" if c.get("page_number") is not None else "]")
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)