"""Online Production Traffic Sampler & Automatic Bad Case Feedback Loop.

Implements the production Observe -> Sample -> Audit -> Bad Case 回流闭环:
  - Samples completed production sessions via deterministic hash sampling (`sample_rate`)
    PLUS 100% anomaly-triggered sampling (failed/budget-exceeded/slow/ungrounded citation).
  - Audits Grounding citations (`verify_grounding`), latency SLO, and optional LLM-as-a-Judge.
  - Automatically funnels detected production Bad Cases into `ControlPlaneStore.bad_cases`
    for regression testing on subsequent Agent releases.
"""

from __future__ import annotations

import hashlib
from typing import Any

from weaverrt.context.engine import verify_grounding
from weaverrt.control_plane.registry import ControlPlaneStore
from weaverrt.domain import EventType, Session, SessionStatus
from weaverrt.runtime.store import RuntimeStore


class OnlineSampler:
    def __init__(
        self,
        runtime_store: RuntimeStore,
        cp_store: ControlPlaneStore,
        *,
        sample_rate: float = 0.20,
        latency_slo_ms: int = 10_000,
    ) -> None:
        self.runtime_store = runtime_store
        self.cp_store = cp_store
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.latency_slo_ms = latency_slo_ms

    def _should_random_sample(self, session_id: str) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
        return bucket < self.sample_rate

    def audit_sessions(self, session_ids: list[str] | None = None) -> dict[str, Any]:
        sessions = (
            [self.runtime_store.get_session(sid) for sid in session_ids]
            if session_ids is not None
            else self.runtime_store.list_sessions(limit=200)
        )
        valid_sessions: list[Session] = [s for s in sessions if s is not None]

        inspected = 0
        sampled_ids: list[str] = []
        bad_case_ids: list[str] = []
        grounding_failures = 0

        for sess in valid_sessions:
            events = self.runtime_store.get_events(sess.session_id)
            if any(e.event_type == EventType.ONLINE_SAMPLED for e in events):
                continue

            dur_ms = max(0, sess.updated_at_ms - sess.created_at_ms)
            # Gather all citations retrieved across turns in this session
            retrieved: list[str] = []
            for ev in events:
                if ev.event_type == EventType.CONTEXT_BUILT:
                    retrieved.extend(ev.payload.get("retrieved_citations") or [])
            grounded_ok, hallucinated = verify_grounding(sess.final_output, retrieved)

            is_anomaly = (
                sess.status in (SessionStatus.FAILED, SessionStatus.BUDGET_EXCEEDED)
                or dur_ms > self.latency_slo_ms
                or not grounded_ok
            )
            if not is_anomaly and not self._should_random_sample(sess.session_id):
                continue

            inspected += 1
            sampled_ids.append(sess.session_id)

            issues: list[str] = []
            root_cause = ""
            if sess.status in (SessionStatus.FAILED, SessionStatus.BUDGET_EXCEEDED):
                issues.append(f"Terminal error status={sess.status.value}: {sess.error}")
                root_cause = "online_runtime_failure"
            if not grounded_ok:
                grounding_failures += 1
                issues.append(f"Hallucinated ungrounded citations: {hallucinated}")
                root_cause = root_cause or "hallucinated_citation"
            if dur_ms > self.latency_slo_ms:
                issues.append(f"Latency SLO breach: {dur_ms}ms > {self.latency_slo_ms}ms")
                root_cause = root_cause or "latency_slo_breach"

            self.runtime_store.append_event(
                sess.session_id,
                sess.current_turn,
                EventType.GROUNDING_CHECKED,
                {"grounded_ok": grounded_ok, "hallucinated": hallucinated, "retrieved_count": len(retrieved)},
            )
            self.runtime_store.append_event(
                sess.session_id,
                sess.current_turn,
                EventType.ONLINE_SAMPLED,
                {"anomaly_triggered": is_anomaly, "issues": issues, "duration_ms": dur_ms},
            )

            if issues:
                dep = self.runtime_store.get_deployment(sess.deployment_id)
                art_id = f"{dep.agent_name}@{dep.agent_version}" if dep else sess.deployment_id
                bcid = self.cp_store.save_bad_case(
                    source_run_id="online_sampler",
                    artifact_id=art_id,
                    case_id=f"prod_{sess.session_id}",
                    category="online_production",
                    root_cause=root_cause,
                    user_input=sess.user_input,
                    actual_output=sess.final_output or sess.error,
                    feedback="; ".join(issues),
                )
                bad_case_ids.append(bcid)

        return {
            "total_candidates": len(valid_sessions),
            "sampled_count": inspected,
            "sampled_session_ids": sampled_ids,
            "bad_cases_created": bad_case_ids,
            "grounding_failures": grounding_failures,
        }
