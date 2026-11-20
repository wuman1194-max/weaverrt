"""End-to-End Interactive Showcase of WeaverRT (M1 -> M6).

Usage:
  python examples/demo_full_lifecycle.py              # Run full M1-M6 benchmark & print report
  python examples/demo_full_lifecycle.py --serve      # Also keep Control Plane Web Console running on :8787
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_m5_m6_control_plane_and_eval import ARTIFACT_V1_YAML, ARTIFACT_V2_YAML, build_eval_dataset
from weaverrt.context.engine import BudgetedContextBuilder, make_read_artifact_tool
from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.control_plane.registry import AgentArtifact, ControlPlaneStore
from weaverrt.control_plane.server import ControlPlaneServer
from weaverrt.domain import ModelResponse, ToolCall
from weaverrt.eval.engine import EvalCase, EvalRunner
from weaverrt.kernel.context import BaselineContextBuilder
from weaverrt.kernel.model import ScriptedModelAdapter
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime
from weaverrt.sandbox.linux_sandbox import SandboxPolicy, WSL2LinuxNamespaceSandbox


def main() -> None:
    ap = argparse.ArgumentParser(description="WeaverRT Full Agent Infra Lifecycle Demo")
    ap.add_argument("--serve", action="store_true", help="Keep Web Control Plane alive on http://127.0.0.1:8787")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    data_dir = ROOT / ".demo_data"
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    rt_store = RuntimeStore(data_dir / "runtime.db", data_dir / "workspaces")
    cp_store = ControlPlaneStore(data_dir / "control_plane.db")
    mem_store = LayeredMemoryStore(data_dir / "memory.db")
    mem_store.ingest_document(
        "agentsphere_slo",
        "Production SLO: P95 < 2500ms, 0 security violations, Exactly-Once idempotency on crash recovery.",
        source="slo_spec.md",
    )

    print("=== [1/4] Linux Namespace Sandbox Security Audit ===")
    sb = WSL2LinuxNamespaceSandbox(audit_log_path=data_dir / "sandbox_audit.jsonl")
    if sb.is_available():
        ws = data_dir / "workspaces" / "sandbox_demo"
        pol = SandboxPolicy(timeout_s=2.0, cpu_limit_s=1, memory_limit_mb=160, allow_network=False)
        for label, code in [
            ("Benign Compute", "print(sum(range(101)))"),
            ("Path Traversal", "open('/etc/passwd').read()"),
            ("Network Exfil ", "import socket; socket.create_connection(('1.1.1.1', 80), 1)"),
            ("CPU Bomb      ", "while True: pass"),
            ("Fork Bomb     ", "import os; os.fork()"),
        ]:
            r = sb.run_python(code, ws, pol)
            print(f"  {label} -> ok={str(r.ok):5s} violation={r.violation or 'NONE':16s} ({r.duration_ms} ms)")
    else:
        print("  (WSL2/Linux unshare not detected, skipped)")

    print("\n=== [2/4] Registering v1 (Naive) & v2 (Budgeted Context + Independent Verifier) ===")
    v1 = cp_store.register_artifact(AgentArtifact.from_yaml(ARTIFACT_V1_YAML), actor="demo")
    v2 = cp_store.register_artifact(AgentArtifact.from_yaml(ARTIFACT_V2_YAML), actor="demo")
    dataset = build_eval_dataset()

    def make_rt(art: AgentArtifact, case: EvalCase) -> ManagedRuntime:
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="fetch_status",
                description="Fetch status",
                parameters={"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
                handler=lambda a, c: f"service={a['service']} state=HEALTHY",
                required_scopes=("read",),
            )
        )
        reg.register(
            ToolSpec(
                name="fetch_huge_log",
                description="Fetch 12KB log",
                parameters={"type": "object", "properties": {"idx": {"type": "integer"}}, "required": ["idx"]},
                handler=lambda a, c: "\n".join(
                    [f"INFO line {k} " + ("x" * 40) for k in range(1, 150)]
                    + [f"FATAL ERROR_CODE=ERR_OOM_{a['idx']}"]
                    + [f"INFO line {k} " + ("x" * 40) for k in range(151, 220)]
                ),
                required_scopes=("read",),
            )
        )
        reg.register(make_read_artifact_tool())

        def handler(messages, tools, role_hint, idx):
            if role_hint == "verifier":
                if "UNVERIFIED_DRAFT" in messages[-1].content:
                    return ModelResponse(content=json.dumps({"passed": False, "feedback": "Fix draft quota"}))
                return ModelResponse(content=json.dumps({"passed": True, "feedback": "OK"}))
            if case.category == "tool_use":
                svc = case.user_input.split("service ")[1].split()[0]
                if not any(m.role == "tool" for m in messages):
                    return ModelResponse(tool_calls=[ToolCall("t1", "fetch_status", {"service": svc})])
                return ModelResponse(content=f"{svc} HEALTHY")
            if case.category == "long_context":
                num = int(case.case_id.split("_")[-1])
                tmsgs = [m for m in messages if m.role == "tool"]
                if not tmsgs:
                    return ModelResponse(tool_calls=[ToolCall("h1", "fetch_huge_log", {"idx": num})])
                if "EXTERNALIZED ARTIFACT" in tmsgs[-1].content and len(tmsgs) == 1:
                    return ModelResponse(
                        tool_calls=[ToolCall("r1", "read_artifact", {"artifact_path": ".artifacts/tool_h1.txt", "start_line": 148, "end_line": 152})]
                    )
                return ModelResponse(content=f"Diagnosed ERR_OOM_{num}")
            num = int(case.case_id.split("_")[-1])
            if not any("Independent Verifier Feedback" in m.content for m in messages):
                return ModelResponse(content=f"UNVERIFIED_DRAFT {num * 99}")
            return ModelResponse(content=f"QUOTA_VERIFIED_{num * 100}")

        ctx_b = (
            BudgetedContextBuilder(max_context_tokens=art.max_context_tokens, max_tool_output_chars=400, memory_store=mem_store)
            if art.context_strategy == "budgeted"
            else BaselineContextBuilder()
        )
        return ManagedRuntime(rt_store, ScriptedModelAdapter(handler=handler), reg, context_builder=ctx_b)

    print("\n=== [3/4] Running 20-Case Eval Benchmark & Release Gate ===")
    runner = EvalRunner(rt_store, cp_store)
    r1 = runner.run_experiment(v1, "agentsphere_bench_v1", dataset, make_rt)
    ok1, msg1 = cp_store.promote_to_release("ops-agent", "1.0.0")
    print(f"  v1.0.0 -> pass={r1['pass_rate']:.1%} ({r1['passed_cases']}/20) avg_tokens={r1['avg_tokens']:.0f} p95={r1['p95_ms']}ms gate={r1['gate_passed']}")
    print(f"           release attempt -> ok={ok1} ({msg1})")
    print(f"           bad cases harvested -> {len(cp_store.list_bad_cases())}")

    r2 = runner.run_experiment(v2, "agentsphere_bench_v1", dataset, make_rt)
    ok2, msg2 = cp_store.promote_to_release("ops-agent", "2.0.0")
    diff = EvalRunner.compare_runs(r1, r2)
    print(f"  v2.0.0 -> pass={r2['pass_rate']:.1%} ({r2['passed_cases']}/20) avg_tokens={r2['avg_tokens']:.0f} p95={r2['p95_ms']}ms gate={r2['gate_passed']}")
    print(f"           release attempt -> ok={ok2} ({msg2})")
    print(f"           regression diff -> pass_delta={diff['pass_rate_delta']:+.1%}, token_delta={diff['avg_tokens_delta_pct']:+.1f}%, improved={len(diff['improved_cases'])}, regressed={len(diff['regressed_cases'])}")

    if args.serve:
        srv = ControlPlaneServer(rt_store, cp_store, host="127.0.0.1", port=args.port)
        url = srv.start()
        print(f"\n=== [4/4] Control Plane Web Console LIVE at {url} (Ctrl+C to stop) ===")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            srv.stop()
    else:
        print("\n=== [4/4] Done! Run with `--serve` to inspect live Web Console at http://127.0.0.1:8787 ===")


if __name__ == "__main__":
    main()
