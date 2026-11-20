"""ContextBuilder abstraction (baseline for M1/M2; M3 adds budget compaction, artifact externalization & grounded RAG)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from weaverrt.domain import Message
from weaverrt.kernel.model import estimate_messages_tokens


@dataclass
class ContextBuildResult:
    messages: list[Message]
    estimated_tokens: int
    compacted: bool = False
    externalized_artifacts: list[str] = field(default_factory=list)
    retrieved_citations: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class ContextBuilder(ABC):
    @abstractmethod
    def build(
        self,
        system_prompt: str,
        history: list[Message],
        *,
        session_id: str,
        workspace_path: str,
        query: str = "",
    ) -> ContextBuildResult:
        raise NotImplementedError


class BaselineContextBuilder(ContextBuilder):
    """Simple pass-through ContextBuilder used as baseline and M1/M2 default."""

    def build(
        self,
        system_prompt: str,
        history: list[Message],
        *,
        session_id: str,
        workspace_path: str,
        query: str = "",
    ) -> ContextBuildResult:
        msgs = [Message(role="system", content=system_prompt)] + list(history)
        return ContextBuildResult(
            messages=msgs,
            estimated_tokens=estimate_messages_tokens(msgs),
            stats={"strategy": "baseline", "message_count": len(msgs)},
        )
