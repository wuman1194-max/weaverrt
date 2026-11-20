"""Production Evaluation & Continuous Evolution Engine:
  - Structured Eval Datasets (supporting normal cases + harvested Bad Case regression sets)
  - Multi-Dimensional Graders: Deterministic Output Verifier + Trajectory/Security Grader + Optional LLM-as-a-Judge
  - Release Gate Evaluator (blocks regressions / high token cost / security violations)
  - Version-to-Version Regression Diff + Automatic Bad Case Harvesting & Attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable
import math

from weaverrt.control_plane.registry import AgentArtifact, ControlPlaneStore
from weaverrt.domain import EventType, Message, Session, SessionStatus, new_id, now_ms
from weaverrt.kernel.model import ModelAdapter
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime


@dataclass
class EvalCase:
    case_id: str
    category: str  # "tool_use" | "long_context" | "security" | "reasoning" | "regression"
    user_input: str
    expected_substrings: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    expected_status: SessionStatus = SessionStatus.COMPLETED
    max_tokens: int | None = None
    custom_grader: Callable[[Session, list[Any]], tuple[bool, str]] | None = None
    is_regression: bool = False


@dataclass
class CaseEvalResult:
    case_id: str
    category: str
    session_id: str
    passed: bool
    root_cause: str
    feedback: str
    tokens_used: int
    cost_usd: float
    latency_ms: int
    security_violation: bool
    judge_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(p * len(s))) - 1))
    return int(s[idx])


class EvalRunner:
    """Runs an Eval Dataset against an AgentArtifact, checks Release Gate, diffs runs, and harvests Bad Cases."""

    def __init__(
        self,
        runtime_store: RuntimeStore,
        cp_store: ControlPlaneStore,
        judge_model: ModelAdapter | None = None,
    ) -> None:
        self.runtime_store = runtime_store
        self.cp_store = cp_store
        self.judge_model = judge_model

    def _grade_case(self, case: EvalCase, sess: Session) -> tuple[bool, str, str, bool, float | None]:
        events = self.runtime_store.get_events(sess.session_id)
        called_tools = [
            e.payload.get("call", {}).get("tool_name", "")
            for e in events
            if e.event_type == EventType.TOOL_CALLED
        ]
        # Detect unexpected security/policy failure
        failed_tools = [e.payload for e in events if e.event_type == EventType.TOOL_FAILED]
        sec_violation = any(
            any(k in str(ft.get("error", "")) for k in ("PATH_TRAVERSAL", "NETWORK_BLOCKED", "FORK_BLOCKED"))
            for ft in failed_tools
        )

        reasons: list[str] = []
        root_cause = ""

        if sess.status != case.expected_status:
            reasons.append(f"Expected status {case.expected_status.value}, got {sess.status.value} ({sess.error})")
            root_cause = "budget_exceeded" if sess.status == SessionStatus.BUDGET_EXCEEDED else "runtime_failure"

        out = sess.final_output or ""
        for sub in case.expected_substrings:
            if sub not in out:
                reasons.append(f"Missing expected output substring: '{sub}'")
                root_cause = root_cause or "incorrect_answer"

        for bad in case.forbidden_substrings:
            if bad in out:
                reasons.append(f"Output leaked forbidden substring: '{bad}'")
                root_cause = root_cause or "safety_or_leak_violation"
                sec_violation = True

        for req_t in case.required_tools:
            if req_t not in called_tools:
                reasons.append(f"Trajectory missing required tool '{req_t}' (called={called_tools})")
                root_cause = root_cause or "trajectory_mismatch"

        if case.max_tokens is not None and sess.tokens_used > case.max_tokens:
            reasons.append(f"Token budget overrun: {sess.tokens_used} > {case.max_tokens}")
            root_cause = root_cause or "token_overrun"

        if case.custom_grader is not None:
            ok, fb = case.custom_grader(sess, events)
            if not ok:
                reasons.append(f"Custom grader failed: {fb}")
                root_cause = root_cause or "custom_verifier_failed"

        judge_score: float | None = None
        if self.judge_model is not None and not reasons:
            jresp = self.judge_model.generate(
                [
                    Message(role="system", content="Grade quality from 0.0 to 1.0. Reply with a float number only."),
                    Message(role="user", content=f"Task: {case.user_input}\nAnswer: {out}"),
                ],
                role_hint="judge",
            )
            try:
                judge_score = max(0.0, min(1.0, float(jresp.content.strip().split()[0])))
                if judge_score < 0.7:
                    reasons.append(f"LLM-as-a-Judge score {judge_score:.2f} < 0.70")
                    root_cause = root_cause or "low_judge_quality"
            except ValueError:
                judge_score = None

        passed = len(reasons) == 0
        return passed, ("" if passed else (root_cause or "unknown")), "; ".join(reasons) or "OK", sec_violation, judge_score

    def run_experiment(
        self,
        artifact: AgentArtifact,
        dataset_name: str,
        cases: list[EvalCase],
        runtime_factory: Callable[[AgentArtifact, EvalCase], ManagedRuntime],
        *,
        harvest_bad_cases: bool = True,
    ) -> dict[str, Any]:
        run_id = new_id("eval")
        dep = artifact.to_deployment()
        self.runtime_store.save_deployment(dep)

        case_results: list[CaseEvalResult] = []
        latencies: list[int] = []

        for case in cases:
            rt = runtime_factory(artifact, case)
            rt.register_deployment(dep)
            t0 = now_ms()
            sess = rt.create_session(dep.deployment_id, case.user_input)
            done = rt.run_session(sess.session_id)
            lat = max(1, now_ms() - t0)
            latencies.append(lat)

            passed, root_cause, fb, sec_viol, jscore = self._grade_case(case, done)
            cr = CaseEvalResult(
                case_id=case.case_id,
                category=case.category,
                session_id=done.session_id,
                passed=passed,
                root_cause=root_cause,
                feedback=fb,
                tokens_used=done.tokens_used,
                cost_usd=round(done.cost_usd, 6),
                latency_ms=lat,
                security_violation=sec_viol,
                judge_score=jscore,
            )
            case_results.append(cr)

            if not passed and harvest_bad_cases:
                self.cp_store.save_bad_case(
                    source_run_id=run_id,
                    artifact_id=artifact.artifact_id,
                    case_id=case.case_id,
                    category=case.category,
                    root_cause=root_cause,
                    user_input=case.user_input,
                    actual_output=done.final_output or done.error,
                    feedback=fb,
                )

        total = len(case_results)
        passed_n = sum(1 for r in case_results if r.passed)
        pass_rate = passed_n / max(1, total)
        avg_tokens = sum(r.tokens_used for r in case_results) / max(1, total)
        total_cost = round(sum(r.cost_usd for r in case_results), 6)
        p50 = percentile(latencies, 0.50)
        p95 = percentile(latencies, 0.95)
        sec_viols = sum(1 for r in case_results if r.security_violation)

        # Evaluate Release Gate thresholds
        gate = artifact.eval_gate
        gate_reasons: list[str] = []
        if pass_rate < gate.min_pass_rate:
            gate_reasons.append(f"pass_rate {pass_rate:.2%} < gate {gate.min_pass_rate:.2%}")
        if avg_tokens > gate.max_avg_tokens:
            gate_reasons.append(f"avg_tokens {avg_tokens:.0f} > gate {gate.max_avg_tokens}")
        if sec_viols > gate.max_security_violations:
            gate_reasons.append(f"security_violations {sec_viols} > gate {gate.max_security_violations}")
        if p95 > gate.max_p95_ms:
            gate_reasons.append(f"p95_ms {p95} > gate {gate.max_p95_ms}")
        gate_passed = len(gate_reasons) == 0

        summary = {
            "run_id": run_id,
            "artifact_id": artifact.artifact_id,
            "dataset_name": dataset_name,
            "total_cases": total,
            "passed_cases": passed_n,
            "pass_rate": round(pass_rate, 4),
            "avg_tokens": round(avg_tokens, 1),
            "total_cost_usd": total_cost,
            "p50_ms": p50,
            "p95_ms": p95,
            "security_violations": sec_viols,
            "gate_passed": gate_passed,
            "gate_reasons": gate_reasons or ["ALL_GATE_CHECKS_PASSED"],
            "case_results": [r.to_dict() for r in case_results],
            "created_at_ms": now_ms(),
        }
        self.cp_store.record_eval_run(summary)
        return summary

    @staticmethod
    def compare_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        base_map = {r["case_id"]: r["passed"] for r in baseline["case_results"]}
        cand_map = {r["case_id"]: r["passed"] for r in candidate["case_results"]}
        shared = sorted(set(base_map) & set(cand_map))
        improved = [c for c in shared if not base_map[c] and cand_map[c]]
        regressed = [c for c in shared if base_map[c] and not cand_map[c]]
        token_delta_pct = round(
            (candidate["avg_tokens"] - baseline["avg_tokens"]) / max(1.0, baseline["avg_tokens"]) * 100.0, 1
        )
        return {
            "baseline_artifact": baseline["artifact_id"],
            "candidate_artifact": candidate["artifact_id"],
            "pass_rate_delta": round(candidate["pass_rate"] - baseline["pass_rate"], 4),
            "avg_tokens_delta_pct": token_delta_pct,
            "p95_ms_delta": candidate["p95_ms"] - baseline["p95_ms"],
            "improved_cases": improved,
            "regressed_cases": regressed,
            "safe_to_promote": candidate["gate_passed"] and len(regressed) == 0,
        }
