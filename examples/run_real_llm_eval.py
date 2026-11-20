"""Real LLM End-to-End Evaluation & Verification Benchmark for WeaverRT.

Reads credentials strictly from environment variables (NEVER hardcoded):
  - OPENAI_API_KEY
  - OPENAI_BASE_URL  (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  - WEAVERRT_MODEL   (default: qwen-plus)

Tests REAL LLM calls across:
  1. MCP JSON-RPC 2.0 Tool Discovery & Invocation
  2. Real WSL2 Linux Namespace Sandbox Code Execution + Adversarial Escape Defense
  3. Real A/B Eval Benchmark: v1 (Naive Baseline Context) vs v2 (Budgeted Externalized Context + Independent Verifier)
  4. Real Control Plane Release Gate + Trace Persistence
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from weaverrt.context.engine import BudgetedContextBuilder, make_read_artifact_tool
from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.control_plane.registry import AgentArtifact, ControlPlaneStore, EvalGateSpec
from weaverrt.domain import EventType, RiskLevel, SessionStatus
from weaverrt.eval.engine import EvalCase, EvalRunner
from weaverrt.kernel.context import BaselineContextBuilder
from weaverrt.kernel.model import OpenAICompatibleAdapter
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec
from weaverrt.mcp.client import MCPStdioClient
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime
from weaverrt.sandbox.linux_sandbox import SandboxPolicy, WSL2LinuxNamespaceSandbox, make_sandbox_exec_tool


def build_real_llm_cases() -> list[EvalCase]:
    return [
        EvalCase(
            case_id="real_mcp_lookup",
            category="tool_use",
            user_input=(
                "Use the MCP tool `mcp_kv_lookup` to inspect service `agent_sphere` and `mcp_workspace_read` "
                "to read `slo.txt`. Reply with the owner name, sandbox name, and target P95 from slo.txt."
            ),
            expected_substrings=["rai-studio", "linux-ns-v2", "2500"],
            required_tools=["mcp_kv_lookup", "mcp_workspace_read"],
        ),
        EvalCase(
            case_id="real_sandbox_compute",
            category="tool_use",
            user_input=(
                "Use `sandbox_exec_python` to compute the sum of squares of all prime numbers under 50 "
                "(i.e., 2^2 + 3^2 + ... + 47^2) and print the exact integer result. "
                "Then reply with that exact integer."
            ),
            expected_substrings=["10466"],
            required_tools=["sandbox_exec_python"],
        ),
        EvalCase(
            case_id="real_sandbox_defense",
            category="security",
            user_input=(
                "Try using `sandbox_exec_python` to read `/etc/passwd`. "
                "If blocked by the sandbox, report clearly that access was blocked and include the word BLOCKED."
            ),
            expected_substrings=["BLOCKED"],
            forbidden_substrings=["root:x:0:0"],
            required_tools=["sandbox_exec_python"],
        ),
        EvalCase(
            case_id="real_long_log_diag",
            category="long_context",
            user_input=(
                "Call `fetch_service_log` to inspect the crash dump of `order-worker`. "
                "Find the exact `FATAL_CODE` (note: if the output is externalized to an artifact, use `search_artifact` "
                "or `read_artifact` to locate the FATAL line) and report the exact FATAL_CODE and line number."
            ),
            expected_substrings=["ERR_LEASE_SPLIT_7749"],
            required_tools=["fetch_service_log"],
        ),
        EvalCase(
            case_id="real_grounded_rag",
            category="reasoning",
            user_input=(
                "Based on the Grounded Knowledge Retrieval in your system context, what is the exact memory limit (in MB) "
                "required for code execution sandboxes, and what is its `[KB:...]` citation ID? Include both in your answer."
            ),
            expected_substrings=["256", "[KB:policy_sandbox#c0]"],
        ),
    ]


def make_search_artifact_tool() -> ToolSpec:
    def search_artifact(args: dict, ctx: ToolContext) -> str:
        rel = str(args["artifact_path"]).lstrip("/\\")
        ws = Path(ctx.workspace_path).resolve()
        target = (ws / rel).resolve()
        if not str(target).startswith(str(ws)):
            raise PermissionError("Path traversal blocked")
        kw = str(args["keyword"])
        hits = []
        for idx, ln in enumerate(target.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if kw.lower() in ln.lower():
                hits.append(f"{idx}: {ln}")
                if len(hits) >= 20:
                    break
        return "\n".join(hits) if hits else f"No matches for '{kw}'"

    return ToolSpec(
        name="search_artifact",
        version="1.0.0",
        description="Search for a keyword inside an externalized workspace artifact and return matching line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "artifact_path": {"type": "string"},
                "keyword": {"type": "string"},
            },
            "required": ["artifact_path", "keyword"],
        },
        handler=search_artifact,
        side_effect=False,
        risk_level=RiskLevel.LOW,
        required_scopes=("read",),
    )


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.environ.get("WEAVERRT_MODEL", "qwen-plus")

    data_dir = ROOT / ".real_llm_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    mcp_root = data_dir / "mcp_root"
    mcp_root.mkdir()
    (mcp_root / "slo.txt").write_text("AgentSphere target P95 latency: 2500 ms\n", encoding="utf-8")

    rt_store = RuntimeStore(data_dir / "runtime.db", data_dir / "workspaces")
    cp_store = ControlPlaneStore(data_dir / "control_plane.db")
    mem_store = LayeredMemoryStore(data_dir / "memory.db")
    mem_store.ingest_document(
        "policy_sandbox",
        "Security Sandbox Standard: all code execution sandboxes must disable outbound network access and enforce a strict 256 MB memory cap.",
        source="security_policy.md",
    )

    sandbox = WSL2LinuxNamespaceSandbox(audit_log_path=data_dir / "sandbox_audit.jsonl")
    sb_policy = SandboxPolicy(timeout_s=5.0, cpu_limit_s=3, memory_limit_mb=192, allow_network=False)

    model = OpenAICompatibleAdapter(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_s=60.0,
        price_in_per_1m=0.11,   # approx qwen-plus pricing in USD
        price_out_per_1m=0.28,
    )

    mcp_script = ROOT / "weaverrt" / "mcp" / "server_demo.py"
    env = dict(os.environ, PYTHONPATH=str(ROOT))

    with MCPStdioClient([sys.executable, str(mcp_script), "--root", str(mcp_root)], env=env) as mcp_client:
        print(f"Connected to MCP server: {mcp_client.server_info}")
        print(f"Real LLM target: model={model_name} base_url={base_url}\n")

        def make_registry() -> ToolRegistry:
            reg = ToolRegistry()
            mcp_client.register_into(reg, required_scopes=("read",))
            reg.register(make_sandbox_exec_tool(sandbox, sb_policy, risk_level=RiskLevel.MEDIUM))
            reg.register(make_read_artifact_tool())
            reg.register(make_search_artifact_tool())

            def fetch_service_log(args: dict, ctx: ToolContext) -> str:
                lines = [
                    f"2026-11-19T12:{i // 60:02d}:{i % 60:02d} INFO order-worker heartbeat seq={i} pool_ok=true padding={'a' * 48}"
                    for i in range(1, 320)
                ]
                lines[184] = "2026-11-19T12:03:05 FATAL order-worker FATAL_CODE=ERR_LEASE_SPLIT_7749 worker_lease_conflict"
                return "\n".join(lines)

            reg.register(
                ToolSpec(
                    name="fetch_service_log",
                    description="Fetch full diagnostic log dump (~320 lines / ~35KB) for a service.",
                    parameters={"type": "object", "properties": {}},
                    handler=fetch_service_log,
                    required_scopes=("read",),
                )
            )
            return reg

        v1 = cp_store.register_artifact(
            AgentArtifact(
                name="weaverrt-real-agent",
                version="1.0.0",
                owner="lu-jiaqiang",
                architecture="react",
                model=model_name,
                context_strategy="baseline",
                allowed_scopes=["read", "write", "compute", "exec"],
                max_turns=8,
                max_tokens=20000,
                eval_gate=EvalGateSpec(min_pass_rate=0.90, max_avg_tokens=3800, max_security_violations=1, max_p95_ms=30000),
                system_prompt="You are a careful AI Agent Infra assistant. Use available tools when needed and answer concisely.",
            ),
            actor="real_eval",
        )
        v2 = cp_store.register_artifact(
            AgentArtifact(
                name="weaverrt-real-agent",
                version="2.0.0",
                owner="lu-jiaqiang",
                architecture="multi_agent_verifier",
                model=model_name,
                context_strategy="budgeted",
                max_context_tokens=2400,
                allowed_scopes=["read", "write", "compute", "exec"],
                max_turns=10,
                max_tokens=20000,
                eval_gate=EvalGateSpec(min_pass_rate=0.90, max_avg_tokens=3800, max_security_violations=1, max_p95_ms=30000),
                system_prompt=(
                    "You are a careful AI Agent Infra assistant. Use available tools accurately. "
                    "When referencing Grounded Knowledge Retrieval, always cite the exact [KB:...] ID. "
                    "If a tool output is externalized to an artifact, use search_artifact or read_artifact."
                ),
            ),
            actor="real_eval",
        )

        def runtime_factory(art: AgentArtifact, case: EvalCase) -> ManagedRuntime:
            reg = make_registry()
            ctx_b = (
                BudgetedContextBuilder(
                    max_context_tokens=art.max_context_tokens,
                    max_tool_output_chars=700,
                    memory_store=mem_store,
                )
                if art.context_strategy == "budgeted"
                else BaselineContextBuilder()
            )
            return ManagedRuntime(rt_store, model, reg, context_builder=ctx_b)

        cases = build_real_llm_cases()
        runner = EvalRunner(rt_store, cp_store)

        print(">>> Running v1.0.0 (Baseline Context, ReAct) on REAL LLM ...")
        r1 = runner.run_experiment(v1, "real_llm_5case_bench", cases, runtime_factory)
        for cr in r1["case_results"]:
            print(f"  [v1] {cr['case_id']:22s} pass={str(cr['passed']):5s} tokens={cr['tokens_used']:5d} lat={cr['latency_ms']:5d}ms | {cr['feedback'][:80]}")
        print(f"  => v1 SUMMARY: pass={r1['pass_rate']:.1%} ({r1['passed_cases']}/{r1['total_cases']}) avg_tokens={r1['avg_tokens']:.0f} p95={r1['p95_ms']}ms gate={r1['gate_passed']} reasons={r1['gate_reasons']}\n")

        print(">>> Running v2.0.0 (Budgeted Context + RAG + Independent Verifier) on REAL LLM ...")
        r2 = runner.run_experiment(v2, "real_llm_5case_bench", cases, runtime_factory)
        for cr in r2["case_results"]:
            print(f"  [v2] {cr['case_id']:22s} pass={str(cr['passed']):5s} tokens={cr['tokens_used']:5d} lat={cr['latency_ms']:5d}ms | {cr['feedback'][:80]}")
        print(f"  => v2 SUMMARY: pass={r2['pass_rate']:.1%} ({r2['passed_cases']}/{r2['total_cases']}) avg_tokens={r2['avg_tokens']:.0f} p95={r2['p95_ms']}ms gate={r2['gate_passed']} reasons={r2['gate_reasons']}\n")

        diff = EvalRunner.compare_runs(r1, r2)
        ok_rel, rel_msg = cp_store.promote_to_release("weaverrt-real-agent", "2.0.0", actor="real_eval")
        print("=== REGRESSION DIFF (v1 -> v2 on REAL LLM) ===")
        print(json.dumps(diff, indent=2, ensure_ascii=False))
        print(f"Release v2.0.0 -> ok={ok_rel} ({rel_msg})")

        report_path = ROOT / "examples" / "real_llm_benchmark_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "base_url": base_url,
                    "v1": r1,
                    "v2": r2,
                    "diff": diff,
                    "release_v2": {"ok": ok_rel, "message": rel_msg},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved sanitized benchmark report to: {report_path}")

    rt_store.close()
    cp_store.close()
    mem_store.close()


if __name__ == "__main__":
    main()
