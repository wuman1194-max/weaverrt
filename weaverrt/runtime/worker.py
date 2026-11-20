"""ManagedRuntime / Worker orchestration: session lifecycle, lease takeover, recovery & approval resume."""

from __future__ import annotations

from typing import Callable

from weaverrt.domain import Deployment, EventType, Session, SessionStatus, new_id
from weaverrt.kernel.approval import PolicyEngine, identity_from_deployment
from weaverrt.kernel.architectures import (
    BudgetExceededError,
    ExecutionContext,
    LeaseLostError,
    get_engine,
    replay_events,
)
from weaverrt.kernel.context import BaselineContextBuilder, ContextBuilder
from weaverrt.kernel.model import ModelAdapter
from weaverrt.kernel.telemetry import Tracer
from weaverrt.kernel.tools import ApprovalRequiredError, ToolExecutor, ToolRegistry
from weaverrt.runtime.store import RuntimeStore


class ManagedRuntime:
    """Production-style Agent Runtime Worker.

    Multiple independent worker processes can point at the same SQLite WAL store.
    If Worker A is SIGKILL'd mid-session, Worker B takes over once the lease expires
    (or on explicit failover), reconstructs exact state from append-only events,
    replays idempotent tool results, and finishes the session cleanly.
    """

    def __init__(
        self,
        store: RuntimeStore,
        model: ModelAdapter,
        registry: ToolRegistry,
        *,
        worker_id: str | None = None,
        context_builder: ContextBuilder | None = None,
        policy: PolicyEngine | None = None,
        secrets: dict[str, str] | None = None,
        verifier_model: ModelAdapter | None = None,
        deterministic_verifier: Callable[[str, str, str], tuple[bool, str]] | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.registry = registry
        self.worker_id = worker_id or new_id("worker")
        self.context_builder = context_builder or BaselineContextBuilder()
        self.policy = policy or PolicyEngine()
        self.secrets = dict(secrets or {})
        self.verifier_model = verifier_model
        self.deterministic_verifier = deterministic_verifier
        self.tool_executor = tool_executor or ToolExecutor(
            registry=registry,
            store=store,
            policy=self.policy,
            secrets=self.secrets,
        )

    def register_deployment(self, dep: Deployment) -> Deployment:
        return self.store.save_deployment(dep)

    def create_session(self, deployment_id: str, user_input: str, session_id: str | None = None) -> Session:
        sess, _ = self.store.create_session(deployment_id, user_input, session_id=session_id)
        return sess

    def run_session(
        self,
        session_id: str,
        *,
        force_takeover: bool = False,
        on_turn_hook: Callable[[int, str], None] | None = None,
    ) -> Session:
        sess = self.store.get_session(session_id)
        if sess is None:
            raise KeyError(f"Session not found: {session_id}")
        dep = self.store.get_deployment(sess.deployment_id)
        if dep is None:
            raise KeyError(f"Deployment not found: {sess.deployment_id}")

        acquired, recovered, prev_worker = self.store.acquire_lease(
            session_id,
            self.worker_id,
            ttl_ms=dep.lease_ttl_ms,
            force_takeover=force_takeover,
        )
        if not acquired:
            raise RuntimeError(
                f"Could not acquire lease for session {session_id} (status={sess.status.value}, holder={prev_worker})"
            )

        sess = self.store.get_session(session_id)
        assert sess is not None
        events = self.store.get_events(session_id)
        replayed = replay_events(sess.user_input, events)
        tracer = Tracer(self.store, session_id)
        identity = identity_from_deployment(dep, session_id)
        engine = get_engine(dep.architecture)

        ctx = ExecutionContext(
            session=sess,
            deployment=dep,
            identity=identity,
            worker_id=self.worker_id,
            store=self.store,
            model=self.model,
            verifier_model=self.verifier_model,
            context_builder=self.context_builder,
            tool_executor=self.tool_executor,
            tracer=tracer,
            deterministic_verifier=self.deterministic_verifier,
            on_turn_hook=on_turn_hook,
        )

        with tracer.span(
            f"session:{dep.architecture}",
            kind="session",
            attributes={"worker_id": self.worker_id, "recovered": recovered, "architecture": dep.architecture},
        ):
            try:
                final_output = engine.run(ctx, replayed)
                self.store.append_event(
                    session_id,
                    ctx.session.current_turn,
                    EventType.SESSION_COMPLETED,
                    {"final_output": final_output[:500]},
                )
                return self.store.update_session_state(
                    session_id,
                    status=SessionStatus.COMPLETED,
                    final_output=final_output,
                    release_lease=True,
                )
            except ApprovalRequiredError as ap:
                return self.store.update_session_state(
                    session_id,
                    status=SessionStatus.SUSPENDED_APPROVAL,
                    error=f"Waiting approval {ap.approval_id}: {ap.reason}",
                    release_lease=True,
                )
            except BudgetExceededError as be:
                return self.store.update_session_state(
                    session_id,
                    status=SessionStatus.BUDGET_EXCEEDED,
                    error=str(be),
                    release_lease=True,
                )
            except LeaseLostError as le:
                return self.store.get_session(session_id) or sess
            except Exception as exc:
                self.store.append_event(
                    session_id,
                    ctx.session.current_turn,
                    EventType.SESSION_FAILED,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return self.store.update_session_state(
                    session_id,
                    status=SessionStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    release_lease=True,
                )
