# from openai import OpenAI
#
#
# class VLLMChat:
#     def __init__(
#         self,
#         model: str,
#         base_url: str = "http://localhost:13001/v1",
#         api_key: str = "EMPTY",
#     ):
#         self.model = model
#         self.client = OpenAI(
#             base_url=base_url,
#             api_key=api_key,
#         )
#
#     def invoke(self, messages, temperature: float = 0.0) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=temperature,
#         )
#         return response.choices[0].message.content

# src/rag_platform/retrieval/LLM/vLLM/vllm_chat.py

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import OpenAI


def _convert_messages(messages: List[BaseMessage]) -> list[dict]:
    converted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "assistant"

        converted.append({
            "role": role,
            "content": msg.content,
        })

    return converted


class VLLMChat(BaseChatModel):
    model_name: str
    base_url: str
    temperature: float = 0.0
    api_key: str = "EMPTY"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float = 0.0,
        api_key: str = "EMPTY"
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            api_key=api_key,
        )
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "vllm-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": _convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if stop:
            payload["stop"] = stop

        response = self._client.chat.completions.create(**payload)
        content = response.choices[0].message.content or ""

        message = AIMessage(content=content)

        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )