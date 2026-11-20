"""Decoupled Execution Architectures built on the same Harness Kernel & Event-Sourced Runtime.

Supports:
  1. ReActEngine              — Thought / ToolCall / Observation loop with full crash-resumable replay
  2. PlannerExecutorEngine    — Explicit persistent Plan -> per-step resumable execution -> Synthesis
  3. MultiAgentVerifierEngine — Solver + Independent Verifier (fresh isolated context + optional deterministic oracle)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import json

from weaverrt.domain import (
    Deployment,
    Event,
    EventType,
    Message,
    ModelResponse,
    Session,
    ToolCall,
    ToolResult,
)
from weaverrt.kernel.approval import AgentIdentity
from weaverrt.kernel.context import ContextBuilder
from weaverrt.kernel.model import ModelAdapter
from weaverrt.kernel.telemetry import Tracer
from weaverrt.kernel.tools import ToolContext, ToolExecutor
from weaverrt.runtime.store import RuntimeStore


class BudgetExceededError(Exception):
    pass


class LeaseLostError(Exception):
    pass


@dataclass
class ReplayedState:
    history: list[Message] = field(default_factory=list)
    completed_turns: int = 0
    inflight_turn: int | None = None
    inflight_assistant_msg: Message | None = None
    inflight_completed_calls: dict[str, ToolResult] = field(default_factory=dict)
    plan_steps: list[str] = field(default_factory=list)
    completed_plan_steps: set[int] = field(default_factory=set)
    verifier_rounds_done: int = 0


def replay_events(user_input: str, events: list[Event]) -> ReplayedState:
    """Reconstruct exact execution state from append-only Event Log.

    Guarantees that after a SIGKILL or SUSPENDED_APPROVAL resume:
      - Already-committed LLM responses in an in-flight turn are NOT re-queried.
      - Already-completed tool calls in that turn are NOT re-executed.
      - Completed planner steps are skipped automatically.
    """
    state = ReplayedState(history=[Message(role="user", content=user_input)])

    # Group turn-level events
    turns_llm: dict[int, Message] = {}
    turns_tools: dict[int, list[ToolResult]] = {}
    completed_turn_indices: set[int] = set()

    for ev in events:
        t = ev.turn_index
        et = ev.event_type
        p = ev.payload
        if et == EventType.PLAN_CREATED:
            state.plan_steps = [str(s) for s in (p.get("steps") or [])]
        elif et == EventType.PLAN_STEP_COMPLETED:
            state.completed_plan_steps.add(int(p["step_index"]))
        elif et == EventType.VERIFIER_CHECKED:
            state.verifier_rounds_done += 1
            if not p.get("passed", False) and p.get("feedback"):
                # Feedback appended after the turn in which verifier checked
                turns_tools.setdefault(t, [])
        elif et == EventType.LLM_RESPONDED:
            role_hint = p.get("role_hint", "default")
            if role_hint in ("planner", "verifier"):
                continue
            tcs = [ToolCall.from_dict(tc) for tc in (p.get("tool_calls") or [])]
            turns_llm[t] = Message(
                role="assistant",
                content=str(p.get("content", "")),
                tool_calls=tcs,
                metadata={"role_hint": role_hint},
            )
        elif et in (EventType.TOOL_SUCCEEDED, EventType.TOOL_FAILED, EventType.TOOL_REPLAYED_IDEMPOTENT):
            tr = ToolResult.from_dict(p)
            turns_tools.setdefault(t, []).append(tr)
        elif et == EventType.TURN_COMPLETED:
            completed_turn_indices.add(t)

    # Rebuild history for all completed turns in order
    for t in sorted(completed_turn_indices):
        if t in turns_llm:
            state.history.append(turns_llm[t])
        seen_calls: set[str] = set()
        for tr in turns_tools.get(t, []):
            if tr.call_id in seen_calls:
                continue
            seen_calls.add(tr.call_id)
            content = tr.output if tr.ok else f"ERROR: {tr.error}"
            state.history.append(
                Message(role="tool", content=content, tool_call_id=tr.call_id, tool_name=tr.tool_name)
            )
        # If verifier rejected after this turn, restore correction message
        for ev in events:
            if ev.turn_index == t and ev.event_type == EventType.VERIFIER_CHECKED and not ev.payload.get("passed"):
                state.history.append(
                    Message(
                        role="user",
                        content=f"[Independent Verifier Feedback] Previous output failed verification: {ev.payload.get('feedback', '')}. Please fix and retry.",
                    )
                )

    state.completed_turns = (max(completed_turn_indices) + 1) if completed_turn_indices else 0

    # Check if there is an in-flight (started/responded but not yet TURN_COMPLETED) turn
    inflight_candidates = [t for t in turns_llm if t not in completed_turn_indices]
    if inflight_candidates:
        t_cur = max(inflight_candidates)
        state.inflight_turn = t_cur
        state.inflight_assistant_msg = turns_llm[t_cur]
        state.inflight_completed_calls = {tr.call_id: tr for tr in turns_tools.get(t_cur, [])}

    return state


@dataclass
class ExecutionContext:
    session: Session
    deployment: Deployment
    identity: AgentIdentity
    worker_id: str
    store: RuntimeStore
    model: ModelAdapter
    context_builder: ContextBuilder
    tool_executor: ToolExecutor
    tracer: Tracer
    verifier_model: ModelAdapter | None = None
    deterministic_verifier: Callable[[str, str, str], tuple[bool, str]] | None = None
    on_turn_hook: Callable[[int, str], None] | None = None

    def check_budget_and_lease(self) -> None:
        fresh = self.store.get_session(self.session.session_id)
        if fresh is not None:
            self.session = fresh
        if self.session.tokens_used >= self.deployment.max_tokens:
            msg = f"Token budget exceeded: {self.session.tokens_used} >= {self.deployment.max_tokens}"
            self.store.append_event(self.session.session_id, self.session.current_turn, EventType.BUDGET_EXCEEDED, {"reason": msg})
            raise BudgetExceededError(msg)
        if self.session.cost_usd >= self.deployment.max_cost_usd:
            msg = f"Cost budget exceeded: ${self.session.cost_usd:.4f} >= ${self.deployment.max_cost_usd:.4f}"
            self.store.append_event(self.session.session_id, self.session.current_turn, EventType.BUDGET_EXCEEDED, {"reason": msg})
            raise BudgetExceededError(msg)
        if not self.store.renew_lease(self.session.session_id, self.worker_id, self.deployment.lease_ttl_ms):
            raise LeaseLostError(f"Worker {self.worker_id} lost lease on {self.session.session_id}")

    def call_llm(
        self,
        turn_index: int,
        system_prompt: str,
        history: list[Message],
        *,
        use_tools: bool = True,
        role_hint: str = "default",
        query: str | None = None,
    ) -> ModelResponse:
        self.check_budget_and_lease()
        effective_query = self.session.user_input if query is None else query
        with self.tracer.span(f"context:{role_hint}", kind="context", attributes={"turn": turn_index}) as cattrs:
            built = self.context_builder.build(
                system_prompt,
                history,
                session_id=self.session.session_id,
                workspace_path=self.session.workspace_path,
                query=effective_query,
            )
            cattrs["estimated_tokens"] = built.estimated_tokens
            cattrs["compacted"] = built.compacted
            cattrs["message_count"] = len(built.messages)
            self.store.append_event(
                self.session.session_id,
                turn_index,
                EventType.CONTEXT_BUILT,
                {
                    "role_hint": role_hint,
                    "estimated_tokens": built.estimated_tokens,
                    "compacted": built.compacted,
                    "externalized_artifacts": built.externalized_artifacts,
                    "retrieved_citations": built.retrieved_citations,
                    "stats": built.stats,
                },
            )
            if built.compacted:
                self.store.append_event(
                    self.session.session_id,
                    turn_index,
                    EventType.COMPACTION_APPLIED,
                    {"stats": built.stats, "externalized_artifacts": built.externalized_artifacts},
                )

        schemas = self.tool_executor.registry.openai_schemas() if use_tools else None
        model = self.verifier_model if (role_hint == "verifier" and self.verifier_model is not None) else self.model
        self.store.append_event(
            self.session.session_id,
            turn_index,
            EventType.LLM_REQUESTED,
            {"role_hint": role_hint, "message_count": len(built.messages), "tool_count": len(schemas or [])},
        )
        with self.tracer.span(f"llm:{role_hint}", kind="llm", attributes={"turn": turn_index}) as lattrs:
            resp = model.generate(built.messages, tools=schemas, role_hint=role_hint)
            lattrs["model"] = resp.model_name
            lattrs["prompt_tokens"] = resp.usage.prompt_tokens
            lattrs["completion_tokens"] = resp.usage.completion_tokens
            lattrs["cost_usd"] = round(resp.usage.cost_usd, 6)
            lattrs["tool_calls"] = len(resp.tool_calls)

        self.session = self.store.update_session_state(
            self.session.session_id,
            current_turn=turn_index,
            add_tokens=resp.usage.total_tokens,
            add_cost_usd=resp.usage.cost_usd,
        )
        payload = resp.to_dict()
        payload["role_hint"] = role_hint
        self.store.append_event(self.session.session_id, turn_index, EventType.LLM_RESPONDED, payload)
        return resp

    def run_single_turn(
        self,
        turn_index: int,
        system_prompt: str,
        history: list[Message],
        *,
        resumed_assistant: Message | None = None,
        resumed_completed_calls: dict[str, ToolResult] | None = None,
        role_hint: str = "default",
        query: str = "",
    ) -> tuple[Message, list[ToolResult]]:
        """Execute (or resume) one atomic Turn: Context -> LLM -> Tools -> TURN_COMPLETED."""
        self.store.upsert_turn(self.session.session_id, turn_index, "RUNNING")
        if resumed_assistant is None:
            self.store.append_event(
                self.session.session_id,
                turn_index,
                EventType.TURN_STARTED,
                {"worker_id": self.worker_id, "role_hint": role_hint},
            )

        with self.tracer.span(f"turn:{turn_index}", kind="turn", attributes={"role_hint": role_hint}) as tattrs:
            if resumed_assistant is not None:
                assistant_msg = resumed_assistant
                tattrs["resumed"] = True
            else:
                resp = self.call_llm(turn_index, system_prompt, history, use_tools=True, role_hint=role_hint, query=query)
                assistant_msg = Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls)
                tattrs["resumed"] = False

            history.append(assistant_msg)
            results: list[ToolResult] = []
            already_done = dict(resumed_completed_calls or {})

            for call in assistant_msg.tool_calls:
                if call.call_id in already_done:
                    tr = already_done[call.call_id]
                else:
                    self.check_budget_and_lease()
                    tctx = ToolContext(
                        session_id=self.session.session_id,
                        turn_index=turn_index,
                        workspace_path=self.session.workspace_path,
                        identity=self.identity,
                    )
                    tr = self.tool_executor.execute(call, tctx, self.tracer)
                results.append(tr)
                content = tr.output if tr.ok else f"ERROR: {tr.error}"
                history.append(Message(role="tool", content=content, tool_call_id=tr.call_id, tool_name=tr.tool_name))

            self.store.upsert_turn(self.session.session_id, turn_index, "COMPLETED")
            self.store.append_event(
                self.session.session_id,
                turn_index,
                EventType.TURN_COMPLETED,
                {"tool_count": len(results), "has_final_answer": len(assistant_msg.tool_calls) == 0},
            )
            if self.on_turn_hook is not None:
                self.on_turn_hook(turn_index, "TURN_COMPLETED")
            return assistant_msg, results


class ExecutionEngine(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, ctx: ExecutionContext, replayed: ReplayedState) -> str:
        raise NotImplementedError


class ReActEngine(ExecutionEngine):
    """Standard iterative Thought -> Tool -> Observation loop with crash/approval resumption."""

    name = "react"

    def run(self, ctx: ExecutionContext, replayed: ReplayedState) -> str:
        history = list(replayed.history)
        turn = replayed.inflight_turn if replayed.inflight_turn is not None else replayed.completed_turns
        resumed_msg = replayed.inflight_assistant_msg
        resumed_calls = replayed.inflight_completed_calls

        while turn < ctx.deployment.max_turns:
            assistant_msg, _ = ctx.run_single_turn(
                turn,
                ctx.deployment.system_prompt,
                history,
                resumed_assistant=resumed_msg,
                resumed_completed_calls=resumed_calls,
                role_hint="react",
            )
            resumed_msg = None
            resumed_calls = None
            if not assistant_msg.tool_calls:
                return assistant_msg.content
            turn += 1

        raise RuntimeError(f"ReActEngine reached max_turns={ctx.deployment.max_turns} without final answer")


class PlannerExecutorEngine(ExecutionEngine):
    """Explicit Plan -> Per-Step Resumable Execution -> Final Synthesis architecture."""

    name = "planner_executor"

    def _parse_plan(self, text: str) -> list[str]:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
                return [str(s).strip() for s in obj["steps"] if str(s).strip()]
            if isinstance(obj, list):
                return [str(s).strip() for s in obj if str(s).strip()]
        except ValueError:
            pass
        lines = [ln.strip(" -*\t0123456789.)") for ln in text.splitlines() if ln.strip()]
        return lines or ["Solve the user request directly"]

    def run(self, ctx: ExecutionContext, replayed: ReplayedState) -> str:
        history = list(replayed.history)
        turn = replayed.inflight_turn if replayed.inflight_turn is not None else replayed.completed_turns
        steps = list(replayed.plan_steps)

        if not steps:
            plan_prompt = (
                ctx.deployment.system_prompt
                + "\nYou are the PLANNER. Break the user task into concise ordered steps as JSON: {\"steps\": [...]}."
            )
            plan_resp = ctx.call_llm(turn, plan_prompt, history, use_tools=False, role_hint="planner")
            steps = self._parse_plan(plan_resp.content)
            ctx.store.append_event(ctx.session.session_id, turn, EventType.PLAN_CREATED, {"steps": steps})

        resumed_msg = replayed.inflight_assistant_msg
        resumed_calls = replayed.inflight_completed_calls

        for idx, step_desc in enumerate(steps):
            if idx in replayed.completed_plan_steps:
                continue
            if resumed_msg is None:
                history.append(
                    Message(role="user", content=f"[Plan Step {idx + 1}/{len(steps)}] Execute: {step_desc}")
                )
            step_done = False
            while turn < ctx.deployment.max_turns:
                assistant_msg, _ = ctx.run_single_turn(
                    turn,
                    ctx.deployment.system_prompt + f"\nCurrent plan step ({idx + 1}/{len(steps)}): {step_desc}",
                    history,
                    resumed_assistant=resumed_msg,
                    resumed_completed_calls=resumed_calls,
                    role_hint="executor",
                    query=step_desc,
                )
                resumed_msg = None
                resumed_calls = None
                turn += 1
                if not assistant_msg.tool_calls:
                    ctx.store.append_event(
                        ctx.session.session_id,
                        turn - 1,
                        EventType.PLAN_STEP_COMPLETED,
                        {"step_index": idx, "step": step_desc, "summary": assistant_msg.content[:300]},
                    )
                    step_done = True
                    break
            if not step_done:
                raise RuntimeError(f"PlannerExecutorEngine exceeded max_turns={ctx.deployment.max_turns}")

        # Final synthesis turn if multiple steps were executed
        if len(steps) > 1 and turn < ctx.deployment.max_turns:
            history.append(
                Message(role="user", content="[Synthesizer] All plan steps completed. Provide the concise final answer.")
            )
            final_msg, _ = ctx.run_single_turn(
                turn,
                ctx.deployment.system_prompt + "\nSynthesize all completed step results into the final response.",
                history,
                role_hint="synthesizer",
            )
            return final_msg.content

        for m in reversed(history):
            if m.role == "assistant" and m.content:
                return m.content
        return ""


class MultiAgentVerifierEngine(ExecutionEngine):
    """Solver + Independent Verifier architecture (directly inspired by SHEIN dual-verifier gate).

    The Verifier runs in a FRESH, ISOLATED context (without the Solver's internal reasoning chain)
    and optionally invokes a deterministic code/rule oracle to eliminate shared-context blind spots.
    """

    name = "multi_agent_verifier"

    def __init__(self, max_verify_rounds: int = 3) -> None:
        self.max_verify_rounds = max_verify_rounds

    @staticmethod
    def _parse_verdict(text: str) -> tuple[bool, str]:
        t = text.strip()
        # Strip optional markdown code fences returned by real LLMs
        if t.startswith("```"):
            lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
            t = "\n".join(lines).strip()
        try:
            d = json.loads(t)
            if isinstance(d, dict) and "passed" in d:
                return bool(d["passed"]), str(d.get("feedback", ""))
        except ValueError:
            pass
        up = t.upper()
        if up.startswith("PASS") or '"PASSED": TRUE' in up or '"PASSED":TRUE' in up:
            return True, t
        return False, t

    @staticmethod
    def _collect_tool_evidence(ctx: ExecutionContext) -> str:
        """Build tamper-proof objective tool evidence from RuntimeStore events (not Solver prose)."""
        lines: list[str] = []
        for ev in ctx.store.get_events(ctx.session.session_id):
            if ev.event_type in (EventType.TOOL_SUCCEEDED, EventType.TOOL_FAILED, EventType.TOOL_REPLAYED_IDEMPOTENT):
                p = ev.payload
                status = "OK" if p.get("ok") else "ERROR"
                body = (p.get("output") if p.get("ok") else p.get("error")) or ""
                if len(body) > 320:
                    body = body[:180] + " ... " + body[-120:]
                lines.append(f"- [turn {ev.turn_index}] {p.get('tool_name')} -> {status}: {body}")
        return "\n".join(lines[-8:])

    def run(self, ctx: ExecutionContext, replayed: ReplayedState) -> str:
        history = list(replayed.history)
        turn = replayed.inflight_turn if replayed.inflight_turn is not None else replayed.completed_turns
        resumed_msg = replayed.inflight_assistant_msg
        resumed_calls = replayed.inflight_completed_calls
        max_rounds = int(ctx.deployment.config.get("max_verify_rounds", self.max_verify_rounds))
        rounds_done = replayed.verifier_rounds_done

        last_candidate = ""
        while rounds_done < max_rounds and turn < ctx.deployment.max_turns:
            # 1. Solver produces candidate (may use tools across turns)
            while turn < ctx.deployment.max_turns:
                assistant_msg, _ = ctx.run_single_turn(
                    turn,
                    ctx.deployment.system_prompt + "\nRole: SOLVER. Produce a verified, accurate solution.",
                    history,
                    resumed_assistant=resumed_msg,
                    resumed_completed_calls=resumed_calls,
                    role_hint="solver",
                )
                resumed_msg = None
                resumed_calls = None
                last_candidate = assistant_msg.content
                last_turn = turn
                turn += 1
                if not assistant_msg.tool_calls:
                    break

            # 2. Independent Verifier (deterministic oracle + evidence-grounded isolated LLM judge)
            with ctx.tracer.span(f"verifier:round_{rounds_done + 1}", kind="verifier") as vattrs:
                det_ok, det_fb = True, ""
                if ctx.deterministic_verifier is not None:
                    det_ok, det_fb = ctx.deterministic_verifier(
                        ctx.session.user_input, last_candidate, ctx.session.workspace_path
                    )
                if not det_ok:
                    passed, feedback = False, f"[DeterministicOracle] {det_fb}"
                else:
                    evidence = self._collect_tool_evidence(ctx)
                    ev_section = f"\n\nObjective Runtime Tool Evidence (ground truth):\n{evidence}" if evidence else ""
                    verifier_msgs = [
                        Message(
                            role="user",
                            content=(
                                f"Task:\n{ctx.session.user_input}"
                                f"{ev_section}\n\n"
                                f"Candidate Solution:\n{last_candidate}\n\n"
                                'Return strict JSON only: {"passed": true/false, "feedback": "..."}'
                            ),
                        )
                    ]
                    vresp = ctx.call_llm(
                        last_turn,
                        (
                            "You are an INDEPENDENT VERIFIER. Audit ONLY whether the Candidate Solution satisfies the Task "
                            "and is consistent with the Objective Runtime Tool Evidence (if any). "
                            "Trust verified sandbox/tool outputs as ground truth; do not reject valid tool outputs or apply unrelated policies."
                        ),
                        verifier_msgs,
                        use_tools=False,
                        role_hint="verifier",
                        query=("" if evidence else ctx.session.user_input),
                    )
                    passed, feedback = self._parse_verdict(vresp.content)

                vattrs["passed"] = passed
                vattrs["feedback"] = feedback[:200]
                ctx.store.append_event(
                    ctx.session.session_id,
                    last_turn,
                    EventType.VERIFIER_CHECKED,
                    {"round": rounds_done + 1, "passed": passed, "feedback": feedback},
                )
            rounds_done += 1

            if passed:
                return last_candidate
            history.append(
                Message(
                    role="user",
                    content=f"[Independent Verifier Feedback] Previous output failed verification: {feedback}. Please fix and retry.",
                )
            )

        return last_candidate


def _workflow_factory() -> ExecutionEngine:
    from weaverrt.kernel.workflow import WorkflowEngine

    return WorkflowEngine()


ENGINES: dict[str, Callable[[], ExecutionEngine]] = {
    "react": ReActEngine,
    "workflow": _workflow_factory,
    "planner_executor": PlannerExecutorEngine,
    "multi_agent_verifier": MultiAgentVerifierEngine,
}


def get_engine(name: str) -> ExecutionEngine:
    factory = ENGINES.get(name)
    if factory is None:
        raise ValueError(f"Unknown architecture '{name}'. Supported: {sorted(ENGINES)}")
    return factory()
