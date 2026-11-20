"""Versioned ToolRegistry and Exactly-Once ToolExecutor with policy, approval, secret redaction & tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from weaverrt.domain import (
    EventType,
    PolicyAction,
    RiskLevel,
    ToolCall,
    ToolResult,
    compute_idempotency_key,
)
from weaverrt.kernel.approval import AgentIdentity, PolicyEngine
from weaverrt.kernel.telemetry import Tracer
from weaverrt.runtime.store import RuntimeStore


class ApprovalRequiredError(Exception):
    """Raised when a tool call requires human approval, cleanly suspending the Session."""

    def __init__(self, approval_id: str, call: ToolCall, reason: str) -> None:
        super().__init__(reason)
        self.approval_id = approval_id
        self.call = call
        self.reason = reason


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    turn_index: int
    workspace_path: str
    identity: AgentIdentity
    secrets: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], str]
    version: str = "1.0.0"
    side_effect: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    required_scopes: tuple[str, ...] = ("read",)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"[v{self.version}] {self.description}",
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Versioned tool registry supporting registration, discovery, and strict schema validation."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, ToolSpec]] = {}  # name -> {version: ToolSpec}
        self._latest: dict[str, str] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools.setdefault(spec.name, {})[spec.version] = spec
        self._latest[spec.name] = spec.version

    def get(self, name: str, version: str | None = None) -> ToolSpec | None:
        versions = self._tools.get(name)
        if not versions:
            return None
        v = version if (version and version in versions) else self._latest[name]
        return versions[v]

    def list_specs(self) -> list[ToolSpec]:
        return [self._tools[n][v] for n, v in sorted(self._latest.items())]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self.list_specs()]

    @staticmethod
    def validate_args(spec: ToolSpec, args: dict[str, Any]) -> str | None:
        if not isinstance(args, dict):
            return "Arguments must be a JSON object"
        schema = spec.parameters or {}
        required = schema.get("required") or []
        for req in required:
            if req not in args:
                return f"Missing required parameter: '{req}'"
        props = schema.get("properties") or {}
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for k, v in args.items():
            if k in props and "type" in props[k]:
                expected = type_map.get(props[k]["type"])
                if expected is not None:
                    # bool is subclass of int in Python; reject bool for integer/number
                    if props[k]["type"] in ("integer", "number") and isinstance(v, bool):
                        return f"Parameter '{k}' must be {props[k]['type']}, got bool"
                    if not isinstance(v, expected):
                        return f"Parameter '{k}' must be {props[k]['type']}, got {type(v).__name__}"
        return None


def redact_secrets(text: str, secrets: dict[str, str]) -> str:
    """Scrub secret values before returning tool outputs to LLM context or logs."""
    out = text
    for name, val in (secrets or {}).items():
        if val and len(val) >= 4 and val in out:
            out = out.replace(val, f"[REDACTED:{name}]")
    return out


class ToolExecutor:
    """Executes tool calls with:
      1. Schema validation
      2. Policy / RBAC & Human-in-the-loop Approval Gate
      3. Deterministic Idempotency Key cache (Exactly-Once side-effect protection)
      4. Secret redaction on output
      5. Structured Event & Trace recording
    """

    def __init__(
        self,
        registry: ToolRegistry,
        store: RuntimeStore,
        policy: PolicyEngine | None = None,
        secrets: dict[str, str] | None = None,
        on_after_idempotency_commit: Callable[[ToolCall, ToolResult], None] | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.policy = policy or PolicyEngine()
        self.secrets = dict(secrets or {})
        # Chaos testing hook (used in M2 crash-window tests)
        self._on_after_idempotency_commit = on_after_idempotency_commit

    def execute(
        self,
        call: ToolCall,
        ctx: ToolContext,
        tracer: Tracer,
    ) -> ToolResult:
        idem_key = compute_idempotency_key(
            session_id=ctx.session_id,
            turn_index=ctx.turn_index,
            call_id=call.call_id,
            tool_name=call.tool_name,
            arguments=call.arguments,
        )
        with tracer.span(
            f"tool:{call.tool_name}",
            kind="tool",
            attributes={"call_id": call.call_id, "idempotency_key": idem_key},
        ) as attrs:
            spec = self.registry.get(call.tool_name, call.tool_version)
            if spec is None:
                res = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    ok=False,
                    output="",
                    error=f"Unknown tool: {call.tool_name}",
                    idempotency_key=idem_key,
                )
                self.store.append_event(ctx.session_id, ctx.turn_index, EventType.TOOL_FAILED, res.to_dict())
                attrs["ok"] = False
                return res

            # 1. Schema validation
            err = self.registry.validate_args(spec, call.arguments)
            if err:
                res = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    ok=False,
                    output="",
                    error=f"Schema validation failed: {err}",
                    idempotency_key=idem_key,
                )
                self.store.append_event(ctx.session_id, ctx.turn_index, EventType.TOOL_FAILED, res.to_dict())
                attrs["ok"] = False
                return res

            # 2. Policy & Approval check
            decision = self.policy.evaluate(ctx.identity, call, spec.required_scopes, spec.risk_level)
            attrs["policy_action"] = decision.action.value
            if decision.action == PolicyAction.DENY:
                res = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    ok=False,
                    output="",
                    error=f"Policy denied ({decision.rule_id}): {decision.reason}",
                    idempotency_key=idem_key,
                )
                self.store.append_event(ctx.session_id, ctx.turn_index, EventType.TOOL_FAILED, res.to_dict())
                attrs["ok"] = False
                return res

            if decision.action == PolicyAction.REQUIRE_APPROVAL:
                existing = self.store.get_approval_for_call(ctx.session_id, call.call_id)
                if existing is None:
                    aid = self.store.create_approval(
                        ctx.session_id,
                        ctx.turn_index,
                        call.call_id,
                        call.tool_name,
                        call.arguments,
                        decision.reason,
                    )
                    self.store.append_event(
                        ctx.session_id,
                        ctx.turn_index,
                        EventType.APPROVAL_REQUESTED,
                        {"approval_id": aid, "call_id": call.call_id, "tool_name": call.tool_name, "reason": decision.reason},
                    )
                    raise ApprovalRequiredError(aid, call, decision.reason)
                if existing["status"] == "PENDING":
                    raise ApprovalRequiredError(existing["approval_id"], call, decision.reason)
                if existing["status"] == "REJECTED":
                    res = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        ok=False,
                        output="",
                        error=f"Human approval rejected by {existing['decided_by']}",
                        idempotency_key=idem_key,
                    )
                    self.store.append_event(ctx.session_id, ctx.turn_index, EventType.TOOL_FAILED, res.to_dict())
                    attrs["ok"] = False
                    return res

            # 3. Exactly-Once Idempotency Cache lookup (critical for crash recovery!)
            cached = self.store.get_idempotent_result(idem_key)
            if cached is not None:
                self.store.append_event(
                    ctx.session_id,
                    ctx.turn_index,
                    EventType.TOOL_REPLAYED_IDEMPOTENT,
                    cached.to_dict(),
                )
                attrs["replayed"] = True
                attrs["ok"] = cached.ok
                return cached

            # 4. Execute tool
            self.store.append_event(
                ctx.session_id,
                ctx.turn_index,
                EventType.TOOL_CALLED,
                {"call": call.to_dict(), "version": spec.version, "idempotency_key": idem_key, "side_effect": spec.side_effect},
            )
            exec_ctx = ToolContext(
                session_id=ctx.session_id,
                turn_index=ctx.turn_index,
                workspace_path=ctx.workspace_path,
                identity=ctx.identity,
                secrets=self.secrets,
            )
            try:
                raw_out = str(spec.handler(call.arguments, exec_ctx))
                safe_out = redact_secrets(raw_out, self.secrets)
                res = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    ok=True,
                    output=safe_out,
                    idempotency_key=idem_key,
                    metadata={"version": spec.version, "side_effect": spec.side_effect},
                )
            except Exception as exc:
                safe_err = redact_secrets(str(exc), self.secrets)
                res = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    ok=False,
                    output="",
                    error=safe_err,
                    idempotency_key=idem_key,
                    metadata={"version": spec.version, "side_effect": spec.side_effect},
                )

            # 5. Commit idempotency record BEFORE emitting completion event (Write-Ahead guarantee)
            self.store.save_idempotent_result(idem_key, ctx.session_id, call.tool_name, call.arguments, res)
            if self._on_after_idempotency_commit is not None:
                self._on_after_idempotency_commit(call, res)

            ev_type = EventType.TOOL_SUCCEEDED if res.ok else EventType.TOOL_FAILED
            self.store.append_event(ctx.session_id, ctx.turn_index, ev_type, res.to_dict())
            attrs["ok"] = res.ok
            attrs["replayed"] = False
            return res
