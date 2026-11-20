"""Standalone child worker process used by M2 OS-level SIGKILL / TerminateProcess chaos tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from weaverrt.domain import ModelResponse, RiskLevel, ToolCall
from weaverrt.kernel.model import ScriptedModelAdapter
from weaverrt.kernel.tools import ToolContext, ToolExecutor, ToolRegistry, ToolSpec
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime


def build_registry(ledger_path: Path) -> ToolRegistry:
    reg = ToolRegistry()

    def charge_ledger(args: dict, ctx: ToolContext) -> str:
        # REAL external side effect outside SQLite: append one line to ledger file
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(f"{args['tx_id']}:{args['amount']}\n")
            f.flush()
        return f"CHARGED:{args['tx_id']}:{args['amount']}"

    def write_step(args: dict, ctx: ToolContext) -> str:
        p = Path(ctx.workspace_path) / f"step_{args['step']}.txt"
        with p.open("a", encoding="utf-8") as f:
            f.write(f"ran:{args['step']}\n")
            f.flush()
        return f"STEP_{args['step']}_DONE"

    reg.register(
        ToolSpec(
            name="charge_ledger",
            description="Append one irreversible charge entry to external ledger",
            parameters={
                "type": "object",
                "properties": {"tx_id": {"type": "string"}, "amount": {"type": "integer"}},
                "required": ["tx_id", "amount"],
            },
            handler=charge_ledger,
            side_effect=True,
            risk_level=RiskLevel.LOW,
            required_scopes=("write",),
        )
    )
    reg.register(
        ToolSpec(
            name="write_step",
            description="Record execution of a plan step",
            parameters={"type": "object", "properties": {"step": {"type": "integer"}}, "required": ["step"]},
            handler=write_step,
            side_effect=True,
            risk_level=RiskLevel.LOW,
            required_scopes=("write",),
        )
    )
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--workspaces", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--scenario", choices=["react_idempotent", "planner_midcrash"], required=True)
    ap.add_argument("--pause-signal", default="")
    ap.add_argument("--pause-at", default="")
    a = ap.parse_args()

    store = RuntimeStore(a.db, a.workspaces)
    reg = build_registry(Path(a.ledger))

    def signal_and_hang() -> None:
        if a.pause_signal:
            Path(a.pause_signal).write_text(f"READY_TO_KILL:{a.worker_id}", encoding="utf-8")
        # Hang until parent OS-kills this process (SIGKILL / TerminateProcess)
        time.sleep(60)

    if a.scenario == "react_idempotent":
        def handler(messages, tools, role_hint, idx):
            # Look at history: if tool result already present, finish; else call charge_ledger
            tool_msgs = [m for m in messages if m.role == "tool"]
            if not tool_msgs:
                return ModelResponse(
                    content="Charging ledger tx_001",
                    tool_calls=[ToolCall("call_tx_1", "charge_ledger", {"tx_id": "tx_001", "amount": 100})],
                )
            return ModelResponse(content=f"COMPLETED_WITH:{tool_msgs[-1].content}")

        on_commit = (lambda call, res: signal_and_hang()) if a.pause_at == "after_idempotency_commit" else None
        executor = ToolExecutor(reg, store, on_after_idempotency_commit=on_commit)
        rt = ManagedRuntime(store, ScriptedModelAdapter(handler=handler), reg, worker_id=a.worker_id, tool_executor=executor)
        sess = rt.run_session(a.session_id)
        print(json.dumps({"status": sess.status.value, "final_output": sess.final_output}))

    elif a.scenario == "planner_midcrash":
        def handler(messages, tools, role_hint, idx):
            if role_hint == "planner":
                return ModelResponse(content=json.dumps({"steps": ["Execute step 0", "Execute step 1", "Execute step 2"]}))
            if role_hint == "executor":
                last_user = [m for m in messages if m.role == "user"][-1].content
                last_msg = messages[-1]
                for k in (0, 1, 2):
                    if f"Execute step {k}" in last_user and last_msg.role != "tool":
                        return ModelResponse(tool_calls=[ToolCall(f"s_{k}", "write_step", {"step": k})])
                return ModelResponse(content=f"ack:{last_msg.content}")
            if role_hint == "synthesizer":
                return ModelResponse(content="ALL_3_STEPS_SYNTHESIZED")
            return ModelResponse(content="ok")

        def turn_hook(turn_idx: int, phase: str) -> None:
            if a.pause_at == "after_step_0" and turn_idx == 0:
                # Wait briefly until PLAN_STEP_COMPLETED is committed, then hang for OS kill
                pass

        # Pause right before Step 1 starts by checking in handler
        orig_handler = handler

        def pausing_handler(messages, tools, role_hint, idx):
            if a.pause_at == "before_step_1" and role_hint == "executor":
                last_user = [m for m in messages if m.role == "user"][-1].content
                if "Execute step 1" in last_user:
                    signal_and_hang()
            return orig_handler(messages, tools, role_hint, idx)

        rt = ManagedRuntime(store, ScriptedModelAdapter(handler=pausing_handler), reg, worker_id=a.worker_id)
        sess = rt.run_session(a.session_id)
        print(json.dumps({"status": sess.status.value, "final_output": sess.final_output}))


if __name__ == "__main__":
    main()
