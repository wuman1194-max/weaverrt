"""Core domain models for WeaverRT.

Explicitly models the production Agent Runtime entities:
  Deployment -> Workspace -> Session -> Turn -> Event (append-only)
plus ToolSpec/ToolCall/ToolResult, PolicyDecision, ModelResponse, and TraceSpan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import hashlib
import json
import time
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_idempotency_key(
    session_id: str,
    turn_index: int,
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    raw = f"{session_id}|{turn_index}|{call_id}|{tool_name}|{canonical_json(arguments)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class SessionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUSPENDED_APPROVAL = "SUSPENDED_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class TurnStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SUSPENDED = "SUSPENDED"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class EventType(str, Enum):
    SESSION_CREATED = "SESSION_CREATED"
    WORKER_LEASE_ACQUIRED = "WORKER_LEASE_ACQUIRED"
    WORKER_CRASH_RECOVERED = "WORKER_CRASH_RECOVERED"
    TURN_STARTED = "TURN_STARTED"
    PLAN_CREATED = "PLAN_CREATED"
    PLAN_STEP_COMPLETED = "PLAN_STEP_COMPLETED"
    WORKFLOW_NODE_COMPLETED = "WORKFLOW_NODE_COMPLETED"
    SKILL_ACTIVATED = "SKILL_ACTIVATED"
    DELEGATION_VERIFIED = "DELEGATION_VERIFIED"
    GROUNDING_CHECKED = "GROUNDING_CHECKED"
    ONLINE_SAMPLED = "ONLINE_SAMPLED"
    CONTEXT_BUILT = "CONTEXT_BUILT"
    COMPACTION_APPLIED = "COMPACTION_APPLIED"
    LLM_REQUESTED = "LLM_REQUESTED"
    LLM_RESPONDED = "LLM_RESPONDED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_RESOLVED = "APPROVAL_RESOLVED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_SUCCEEDED = "TOOL_SUCCEEDED"
    TOOL_FAILED = "TOOL_FAILED"
    TOOL_REPLAYED_IDEMPOTENT = "TOOL_REPLAYED_IDEMPOTENT"
    VERIFIER_CHECKED = "VERIFIER_CHECKED"
    TURN_COMPLETED = "TURN_COMPLETED"
    SESSION_COMPLETED = "SESSION_COMPLETED"
    SESSION_FAILED = "SESSION_FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    tool_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(
            call_id=str(d["call_id"]),
            tool_name=str(d["tool_name"]),
            arguments=dict(d.get("arguments") or {}),
            tool_version=str(d.get("tool_version", "1.0.0")),
        )


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    ok: bool
    output: str
    error: str = ""
    replayed: bool = False
    idempotency_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolResult":
        return cls(
            call_id=str(d["call_id"]),
            tool_name=str(d["tool_name"]),
            ok=bool(d["ok"]),
            output=str(d.get("output", "")),
            error=str(d.get("error", "")),
            replayed=bool(d.get("replayed", False)),
            idempotency_key=str(d.get("idempotency_key", "")),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=str(d["role"]),
            content=str(d.get("content", "")),
            tool_calls=[ToolCall.from_dict(tc) for tc in (d.get("tool_calls") or [])],
            tool_call_id=str(d.get("tool_call_id", "")),
            tool_name=str(d.get("tool_name", "")),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model_name: str = ""
    finish_reason: str = "stop"
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage.to_dict(),
            "model_name": self.model_name,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str = ""
    rule_id: str = ""


@dataclass
class Deployment:
    deployment_id: str
    agent_name: str
    agent_version: str
    architecture: str = "react"  # "react" | "planner_executor" | "multi_agent_verifier"
    max_turns: int = 12
    max_tokens: int = 32000
    max_cost_usd: float = 1.0
    lease_ttl_ms: int = 5000
    allowed_scopes: list[str] = field(default_factory=lambda: ["read", "write", "compute"])
    system_prompt: str = "You are a reliable production AI agent."
    config: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Deployment":
        return cls(**d)


@dataclass
class Workspace:
    workspace_id: str
    session_id: str
    root_path: str
    created_at_ms: int = field(default_factory=now_ms)


@dataclass
class Session:
    session_id: str
    deployment_id: str
    workspace_path: str
    user_input: str
    status: SessionStatus = SessionStatus.PENDING
    worker_id: str = ""
    lease_expires_at_ms: int = 0
    current_turn: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    final_output: str = ""
    error: str = ""
    version: int = 0
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)


@dataclass(frozen=True)
class Event:
    seq: int
    session_id: str
    turn_index: int
    event_type: EventType
    payload: dict[str, Any]
    created_at_ms: int


@dataclass(frozen=True)
class TraceSpan:
    span_id: str
    parent_span_id: str
    session_id: str
    name: str
    kind: str  # "session" | "turn" | "context" | "llm" | "tool" | "verifier" | "sandbox"
    start_ms: int
    end_ms: int
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
