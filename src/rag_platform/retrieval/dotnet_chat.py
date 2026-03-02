# src/rag_platform/retrieval/dotnet_chat.py
from __future__ import annotations

import re
from typing import Any, List, Optional, LiteralString

from pydantic import PrivateAttr
from pydantic import ConfigDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag_platform.retrieval.dotnet_llm import DotNetLlmClient


_THINK_BLOCK_RE = re.compile(r"(?is)<think>\s*(.*?)\s*</think>\s*(.*)$")
_THINK_CLOSE_ONLY_RE = re.compile(r"(?is)^(.*?)\s*</think>\s*(.*)$")


class DotNetChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper around your .NET /SendGenerateRequest API.

    Returns AIMessage where:
      - content = assistantResponse with <think> removed
      - additional_kwargs["reasoning_content"] = extracted think (if present)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _client: DotNetLlmClient = PrivateAttr()

    def __init__(self, client: DotNetLlmClient, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "dotnet-llm"

    def _extract_think(self, text: str) -> tuple[None, str] | tuple[LiteralString | None, LiteralString]:
        t = (text or "").strip()
        if not t:
            return None, ""

        # Case 1: has <think> ... </think>
        m = _THINK_BLOCK_RE.match(t)
        if m:
            reasoning = m.group(1).strip() or None
            answer = m.group(2).strip()
            return reasoning, answer

        # Case 2: has only </think>
        m = _THINK_CLOSE_ONLY_RE.match(t)
        if m:
            reasoning = m.group(1).strip() or None
            answer = m.group(2).strip()
            return reasoning, answer

        # Case 3: no think tags
        return None, t

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        # Convert chat messages -> prompt
        prompt = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

        resp = self._client.send_generate_request(
            prompt_text=prompt,
            request_text=messages[-1].content if messages else "",
        )

        raw = (resp.get("assistantResponse") or "").strip()
        reasoning, answer = self._extract_think(raw)

        # (Optional) apply stop tokens to final answer only
        if stop:
            for s in stop:
                if s and s in answer:
                    answer = answer.split(s, 1)[0].rstrip()

        msg = AIMessage(
            content=answer,
            additional_kwargs={"reasoning_content": reasoning} if reasoning else {},
        )

        return ChatResult(generations=[ChatGeneration(message=msg)])