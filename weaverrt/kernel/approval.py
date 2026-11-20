"""Authorization PolicyEngine & ApprovalHook for safe tool execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Any

from weaverrt.domain import Deployment, PolicyAction, PolicyDecision, RiskLevel, ToolCall, canonical_json, now_ms


@dataclass(frozen=True)
class AgentIdentity:
    agent_name: str
    agent_version: str
    session_id: str
    allowed_scopes: tuple[str, ...]
    delegation_token: "DelegationToken | None" = None


@dataclass(frozen=True)
class DelegationToken:
    """Short-lived, HMAC-signed, scope-attenuated capability token for Agent-to-Agent delegation.

    Security invariants enforced:
      1. Scope Attenuation: child delegated scopes MUST be a strict subset of parent issuer scopes.
      2. Depth Cap: delegation chain cannot exceed max_depth (prevents unbounded delegation).
      3. Expiry (TTL): expired tokens are deterministically denied.
      4. Tamper-Evident Signature: HMAC-SHA256 verified with runtime signing key.
    """

    issuer_agent: str
    delegate_agent: str
    session_id: str
    delegated_scopes: tuple[str, ...]
    expires_at_ms: int
    depth: int
    max_depth: int
    signature: str

    @staticmethod
    def _sign(payload: dict[str, Any], signing_key: str) -> str:
        msg = canonical_json(payload).encode("utf-8")
        return hmac.new(signing_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    @classmethod
    def issue(
        cls,
        issuer: AgentIdentity,
        delegate_agent: str,
        requested_scopes: tuple[str, ...],
        *,
        ttl_ms: int = 30_000,
        max_depth: int = 2,
        signing_key: str = "weaverrt-runtime-signing-key",
    ) -> "DelegationToken":
        # Enforce scope attenuation: issuer CANNOT delegate scopes it does not possess!
        parent_scopes = set(issuer.allowed_scopes)
        if issuer.delegation_token is not None:
            parent_scopes &= set(issuer.delegation_token.delegated_scopes)
        escalated = [s for s in requested_scopes if s not in parent_scopes]
        if escalated:
            raise PermissionError(
                f"Privilege escalation blocked: '{issuer.agent_name}' cannot delegate scopes {escalated}"
            )
        next_depth = (issuer.delegation_token.depth + 1) if issuer.delegation_token is not None else 1
        eff_max_depth = min(max_depth, issuer.delegation_token.max_depth) if issuer.delegation_token is not None else max_depth
        if next_depth > eff_max_depth:
            raise PermissionError(f"Delegation depth limit exceeded: {next_depth} > {eff_max_depth}")

        exp = now_ms() + ttl_ms
        if issuer.delegation_token is not None:
            exp = min(exp, issuer.delegation_token.expires_at_ms)

        payload = {
            "issuer_agent": issuer.agent_name,
            "delegate_agent": delegate_agent,
            "session_id": issuer.session_id,
            "delegated_scopes": list(requested_scopes),
            "expires_at_ms": exp,
            "depth": next_depth,
            "max_depth": eff_max_depth,
        }
        sig = cls._sign(payload, signing_key)
        return cls(
            issuer_agent=issuer.agent_name,
            delegate_agent=delegate_agent,
            session_id=issuer.session_id,
            delegated_scopes=tuple(requested_scopes),
            expires_at_ms=exp,
            depth=next_depth,
            max_depth=eff_max_depth,
            signature=sig,
        )

    def verify(self, signing_key: str = "weaverrt-runtime-signing-key", *, current_ms: int | None = None) -> tuple[bool, str]:
        ts = now_ms() if current_ms is None else current_ms
        if ts > self.expires_at_ms:
            return False, "Delegation token expired"
        if self.depth > self.max_depth:
            return False, "Delegation depth exceeded"
        payload = {
            "issuer_agent": self.issuer_agent,
            "delegate_agent": self.delegate_agent,
            "session_id": self.session_id,
            "delegated_scopes": list(self.delegated_scopes),
            "expires_at_ms": self.expires_at_ms,
            "depth": self.depth,
            "max_depth": self.max_depth,
        }
        expected = self._sign(payload, signing_key)
        if not hmac.compare_digest(expected, self.signature):
            return False, "Delegation token signature invalid (tampered)"
        return True, "ok"


class PolicyEngine:
    """Evaluates whether an AgentIdentity is allowed to execute a ToolCall.

    Enforces:
      1. Explicit deny list (hard safety guardrail)
      2. DelegationToken signature, TTL expiry, and scope attenuation (if delegated)
      3. Required RBAC scopes (least privilege delegation)
      4. High-risk tool Human-in-the-loop Approval Gate
    """

    def __init__(
        self,
        denied_tools: set[str] | None = None,
        require_approval_for_high_risk: bool = True,
        require_approval_tools: set[str] | None = None,
        signing_key: str = "weaverrt-runtime-signing-key",
    ) -> None:
        self.denied_tools = set(denied_tools or ())
        self.require_approval_for_high_risk = require_approval_for_high_risk
        self.require_approval_tools = set(require_approval_tools or ())
        self.signing_key = signing_key

    def evaluate(
        self,
        identity: AgentIdentity,
        call: ToolCall,
        required_scopes: tuple[str, ...],
        risk_level: RiskLevel,
    ) -> PolicyDecision:
        if call.tool_name in self.denied_tools:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason=f"Tool '{call.tool_name}' is blocked by hard deny policy",
                rule_id="policy.hard_deny",
            )
        effective_scopes = set(identity.allowed_scopes)
        if identity.delegation_token is not None:
            ok, why = identity.delegation_token.verify(self.signing_key)
            if not ok:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason=f"Delegation denied: {why}",
                    rule_id="policy.delegation_invalid",
                )
            effective_scopes &= set(identity.delegation_token.delegated_scopes)

        missing = [s for s in required_scopes if s not in effective_scopes]
        if missing:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason=f"Agent '{identity.agent_name}' lacks required scopes: {missing}",
                rule_id="policy.rbac_scope_missing",
            )
        if call.tool_name in self.require_approval_tools or (
            self.require_approval_for_high_risk and risk_level == RiskLevel.HIGH
        ):
            return PolicyDecision(
                action=PolicyAction.REQUIRE_APPROVAL,
                reason=f"Tool '{call.tool_name}' requires human approval (risk={risk_level.value})",
                rule_id="policy.high_risk_approval",
            )
        return PolicyDecision(action=PolicyAction.ALLOW, reason="allowed", rule_id="policy.allow")


def identity_from_deployment(dep: Deployment, session_id: str) -> AgentIdentity:
    return AgentIdentity(
        agent_name=dep.agent_name,
        agent_version=dep.agent_version,
        session_id=session_id,
        allowed_scopes=tuple(dep.allowed_scopes),
    )
