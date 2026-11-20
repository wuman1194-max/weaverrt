"""Declarative JSON-Serializable DAG Workflow Execution Engine.

Runs on the SAME Harness Kernel & SQLite Event Store as ReAct / Planner-Executor / Multi-Agent:
  - 100% declarative JSON-serializable node specs (`to_dict` / `from_dict`) so ANY remote Worker
    process can reload a Deployment's Workflow DAG straight from SQLite and resume after a crash.
  - Supports Sequential steps, Parallel fan-out / fan-in waves, and Declarative Conditional Routing.
  - Every completed DAG node commits `WORKFLOW_NODE_COMPLETED` atomically into the Event Log.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any

from weaverrt.domain import EventType, Message, ToolCall
from weaverrt.kernel.architectures import ExecutionContext, ExecutionEngine, ReplayedState
from weaverrt.kernel.tools import ToolContext


@dataclass
class WorkflowNode:
    node_id: str
    kind: str  # "tool" | "llm"
    depends_on: list[str] = field(default_factory=list)
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    role_hint: str = "workflow"
    # Declarative condition: {"node": "fetch_a", "contains": "ERROR"} or {"node": "fetch_a", "not_contains": "ERROR"}
    condition: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowNode":
        return cls(
            node_id=str(d["node_id"]),
            kind=str(d["kind"]),
            depends_on=list(d.get("depends_on") or []),
            tool_name=str(d.get("tool_name", "")),
            tool_args=dict(d.get("tool_args") or {}),
            prompt_template=str(d.get("prompt_template", "")),
            role_hint=str(d.get("role_hint", "workflow")),
            condition=dict(d["condition"]) if d.get("condition") else None,
        )


def _eval_condition(cond: dict[str, str] | None, outputs: dict[str, str]) -> bool:
    if not cond:
        return True
    target = outputs.get(str(cond.get("node", "")), "")
    if "contains" in cond:
        return str(cond["contains"]) in target
    if "not_contains" in cond:
        return str(cond["not_contains"]) not in target
    if "equals" in cond:
        return target.strip() == str(cond["equals"]).strip()
    return True


def _render_args(args: dict[str, Any], variables: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str):
            try:
                out[k] = v.format(**variables)
            except KeyError:
                out[k] = v
        else:
            out[k] = v
    return out


class WorkflowEngine(ExecutionEngine):
    """DAG Workflow Engine supporting parallel waves, conditional skipping, and crash-resumable checkpoints."""

    name = "workflow"

    def __init__(self, max_parallel: int = 4) -> None:
        self.max_parallel = max_parallel

    @staticmethod
    def _replay_node_outputs(ctx: ExecutionContext) -> dict[str, str]:
        done: dict[str, str] = {}
        for ev in ctx.store.get_events(ctx.session.session_id):
            if ev.event_type == EventType.WORKFLOW_NODE_COMPLETED:
                done[str(ev.payload["node_id"])] = str(ev.payload.get("output", ""))
        return done

    def _run_node(self, node: WorkflowNode, turn_idx: int, outputs: dict[str, str], ctx: ExecutionContext) -> str:
        variables = dict(outputs, user_input=ctx.session.user_input)
        with ctx.tracer.span(f"workflow_node:{node.node_id}", kind="turn", attributes={"node_kind": node.kind}) as attrs:
            if not _eval_condition(node.condition, outputs):
                attrs["skipped"] = True
                return "[SKIPPED_BY_CONDITION]"

            if node.kind == "tool":
                args = _render_args(node.tool_args, variables)
                tctx = ToolContext(
                    session_id=ctx.session.session_id,
                    turn_index=turn_idx,
                    workspace_path=ctx.session.workspace_path,
                    identity=ctx.identity,
                )
                call = ToolCall(call_id=f"wf_{node.node_id}", tool_name=node.tool_name, arguments=args)
                res = ctx.tool_executor.execute(call, tctx, ctx.tracer)
                if not res.ok:
                    raise RuntimeError(f"Workflow node '{node.node_id}' tool failed: {res.error}")
                return res.output

            if node.kind == "llm":
                rendered = node.prompt_template.format(**variables)
                resp = ctx.call_llm(
                    turn_idx,
                    ctx.deployment.system_prompt + f"\nWorkflow Node: {node.node_id}",
                    [Message(role="user", content=rendered)],
                    use_tools=False,
                    role_hint=node.role_hint,
                    query=rendered,
                )
                return resp.content

            raise ValueError(f"Unsupported workflow node kind: {node.kind}")

    def run(self, ctx: ExecutionContext, replayed: ReplayedState) -> str:
        raw_nodes = ctx.deployment.config.get("workflow_nodes") or []
        if not raw_nodes:
            raise ValueError("WorkflowEngine requires 'workflow_nodes' in deployment.config")
        nodes = [n if isinstance(n, WorkflowNode) else WorkflowNode.from_dict(n) for n in raw_nodes]
        node_map = {n.node_id: n for n in nodes}
        outputs = self._replay_node_outputs(ctx)
        turn_idx = len(outputs)
        last_output = ""

        # Topological wave execution: all ready nodes in a wave execute in parallel!
        while len(outputs) < len(node_map):
            ready = [
                n for nid, n in node_map.items()
                if nid not in outputs and all(dep in outputs for dep in n.depends_on)
            ]
            if not ready:
                remaining = sorted(set(node_map) - set(outputs))
                raise RuntimeError(f"Workflow DAG cycle or missing dependency detected: {remaining}")

            ctx.check_budget_and_lease()
            if len(ready) == 1:
                n = ready[0]
                out = self._run_node(n, turn_idx, dict(outputs), ctx)
                outputs[n.node_id] = out
                ctx.store.append_event(
                    ctx.session.session_id,
                    turn_idx,
                    EventType.WORKFLOW_NODE_COMPLETED,
                    {"node_id": n.node_id, "kind": n.kind, "output": out[:1000], "parallel_wave_size": 1},
                )
                last_output = out
                turn_idx += 1
            else:
                wave_snapshot = dict(outputs)
                wave_results: dict[str, tuple[int, str]] = {}
                with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(ready))) as pool:
                    fut_map = {
                        pool.submit(self._run_node, n, turn_idx + i, wave_snapshot, ctx): (n, turn_idx + i)
                        for i, n in enumerate(ready)
                    }
                    for fut in as_completed(fut_map):
                        n, tidx = fut_map[fut]
                        wave_results[n.node_id] = (tidx, fut.result())

                for n in ready:
                    tidx, out = wave_results[n.node_id]
                    outputs[n.node_id] = out
                    ctx.store.append_event(
                        ctx.session.session_id,
                        tidx,
                        EventType.WORKFLOW_NODE_COMPLETED,
                        {"node_id": n.node_id, "kind": n.kind, "output": out[:1000], "parallel_wave_size": len(ready)},
                    )
                    last_output = out
                turn_idx += len(ready)

        return last_output
