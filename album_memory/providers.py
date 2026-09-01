from __future__ import annotations

import json
import re
from typing import Protocol

from album_memory.config import LLMConfig


class ChatProvider(Protocol):
    def summarize_event(self, evidence: str) -> str: ...

    def classify_conflict(self, statements: list[str], evidence: str) -> dict: ...


class NullChatProvider:
    def summarize_event(self, evidence: str) -> str:
        return ""

    def classify_conflict(self, statements: list[str], evidence: str) -> dict:
        return {"verdict": "unresolved", "reason": "LLM disabled"}


class OpenAICompatibleChatProvider:
    def __init__(self, config: LLMConfig):
        resolved = config.resolved()
        if resolved is None:
            raise ValueError("LLM configuration is disabled")
        base_url, api_key, model = resolved
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=config.timeout_seconds,
        )
        self.model = model

    def _complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是可审计的相册记忆整理器。只依据输入证据，"
                        "禁止推断身份、关系、所有权、职业、心理健康或精确位置。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def summarize_event(self, evidence: str) -> str:
        return self._complete(
            "把以下同一候选事件的事实整理为2到4句中性摘要，保留不确定性和时间边界：\n"
            + evidence
        ).strip()[:2000]

    def classify_conflict(self, statements: list[str], evidence: str) -> dict:
        prompt = (
            "判断结论是 conflict、evolution、coexistence 或 unresolved。"
            "只输出JSON对象，包含 verdict 和 reason；不得激活或删除结论。\n"
            f"结论：{json.dumps(statements, ensure_ascii=False)}\n证据：{evidence}"
        )
        raw = self._complete(prompt)
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {"verdict": "unresolved", "reason": "invalid LLM output"}
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"verdict": "unresolved", "reason": "invalid LLM JSON"}
        if result.get("verdict") not in {"conflict", "evolution", "coexistence", "unresolved"}:
            result["verdict"] = "unresolved"
        return result


def build_chat_provider(config: LLMConfig) -> ChatProvider:
    if not config.enabled:
        return NullChatProvider()
    return OpenAICompatibleChatProvider(config)
