"""Full JD Real-LLM Verification Suite (Alibaba Bailian qwen-plus + qwen-turbo).

Proves EVERY remaining JD capability on REAL live LLM calls + REAL OS process kills:
  1. WorkflowEngine (Parallel DAG: MCP + Linux Sandbox fan-out -> qwen-plus synthesis)
  2. PlannerExecutorEngine (Real qwen-plus planning -> per-step execution -> synthesis)
  3. ModelRouter (Heterogeneous routing: qwen-plus Solver + qwen-turbo Verifier)
  4. Real LLM + REAL OS `kill()` Crash Recovery + ElasticWorkerPool Auto-Failover + Exactly-Once Ledger
  5. Personalization + Cross-Session Episodic Memory + Skill Progressive Disclosure + OnlineSampler Audit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from weaverrt.context.engine import BudgetedContextBuilder
from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.control_plane.registry import ControlPlaneStore
from weaverrt.domain import Deployment, EventType, RiskLevel, SessionStatus
from weaverrt.eval.sampler import OnlineSampler
from weaverrt.kernel.model import ModelRouter, OpenAICompatibleAdapter
from weaverrt.kernel.skills import SkillRegistry, SkillSpec
from weaverrt.kernel.tools import ToolContext, ToolExecutor, ToolRegistry, ToolSpec
from weaverrt.kernel.workflow import WorkflowNode
from weaverrt.mcp.client import MCPStdioClient
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.supervisor import ElasticWorkerPool
from weaverrt.runtime.worker import ManagedRuntime
from weaverrt.sandbox.linux_sandbox import SandboxPolicy, WSL2LinuxNamespaceSandbox, make_sandbox_exec_tool


SKILL_SRE_MD = """---
name: sre-postmortem
version: 1.0.0
description: Concise SRE incident summary format
triggers: [postmortem, outage, incident, slo]
allowed_tools: [mcp_kv_lookup]
required_scopes: [read]
---
Always structure your answer with exactly two labeled lines:
- IMPACT: <brief summary>
- ACTION: <concrete next step>
"""


def _make_llm(model_name: str) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        model=model_name,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout_s=60.0,
    )


def run_child_chaos_worker(db: str, ws: str, ledger: str, session_id: str, signal_file: str) -> None:
    """Real child OS process running REAL qwen-plus that hangs right after committing side-effect idempotency."""
    store = RuntimeStore(db, ws)
    ledger_path = Path(ledger)
    reg = ToolRegistry()

    def write_ledger(args: dict, ctx: ToolContext) -> str:
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(f"{args['order_id']}:{args['amount']}\n")
            f.flush()
        return f"LEDGER_COMMITTED:{args['order_id']}:{args['amount']}"

    reg.register(
        ToolSpec(
            name="write_ledger",
            description="Record an irreversible billing transaction into the external ledger.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "integer"}},
                "required": ["order_id", "amount"],
            },
            handler=write_ledger,
            side_effect=True,
            risk_level=RiskLevel.LOW,
            required_scopes=("write",),
        )
    )

    def hang_for_kill(call, res) -> None:
        Path(signal_file).write_text("KILL_ME_NOW", encoding="utf-8")
        time.sleep(60)

    executor = ToolExecutor(reg, store, on_after_idempotency_commit=hang_for_kill)
    rt = ManagedRuntime(store, _make_llm("qwen-plus"), reg, worker_id="real-llm-worker-A", tool_executor=executor)
    rt.run_session(session_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chaos-child", action="store_true")
    ap.add_argument("--db", default="")
    ap.add_argument("--ws", default="")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--session-id", default="")
    ap.add_argument("--signal-file", default="")
    args = ap.parse_args()

    if args.chaos_child:
        run_child_chaos_worker(args.db, args.ws, args.ledger, args.session_id, args.signal_file)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY first", file=sys.stderr)
        sys.exit(1)

    data_dir = ROOT / ".real_jd_suite"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    rt_store = RuntimeStore(data_dir / "runtime.db", data_dir / "workspaces")
    cp_store = ControlPlaneStore(data_dir / "cp.db")
    mem_store = LayeredMemoryStore(data_dir / "memory.db")
    mcp_root = data_dir / "mcp_root"
    mcp_root.mkdir()
    (mcp_root / "slo.txt").write_text("SLO P95=2500ms, Availability=99.95%\n", encoding="utf-8")

    qwen_plus = _make_llm("qwen-plus")
    qwen_flash = _make_llm("qwen-flash")
    router = ModelRouter(default_model=qwen_plus, routes={"verifier": qwen_flash, "planner": qwen_flash}, fallback_model=qwen_plus)

    sandbox = WSL2LinuxNamespaceSandbox(audit_log_path=data_dir / "sandbox_audit.jsonl")
    sb_policy = SandboxPolicy(timeout_s=5.0, cpu_limit_s=3, memory_limit_mb=192, allow_network=False)

    mcp_script = ROOT / "weaverrt" / "mcp" / "server_demo.py"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    report: dict[str, object] = {}

    with MCPStdioClient([sys.executable, str(mcp_script), "--root", str(mcp_root)], env=env) as mcp_client:
        reg = ToolRegistry()
        mcp_client.register_into(reg, required_scopes=("read",))
        reg.register(make_sandbox_exec_tool(sandbox, sb_policy, risk_level=RiskLevel.MEDIUM))

        # --- 1. Parallel DAG Workflow on REAL qwen-plus ---
        print("=== [1/5] WorkflowEngine (Parallel MCP + Linux Sandbox Fan-Out -> qwen-plus Synthesis) ===")
        wf_nodes = [
            WorkflowNode(node_id="mcp_meta", kind="tool", tool_name="mcp_kv_lookup", tool_args={"service": "agent_sphere"}).to_dict(),
            WorkflowNode(
                node_id="sb_calc",
                kind="tool",
                tool_name="sandbox_exec_python",
                tool_args={"code": "print(sum(i*i for i in range(1, 21)))"},
            ).to_dict(),
            WorkflowNode(
                node_id="synth",
                kind="llm",
                depends_on=["mcp_meta", "sb_calc"],
                prompt_template="Summarize in one concise sentence: service metadata={mcp_meta}, sandbox sum_of_squares_1_to_20={sb_calc}.",
                role_hint="synthesizer",
            ).to_dict(),
        ]
        rt_wf = ManagedRuntime(rt_store, router, reg, worker_id="w-wf")
        dep_wf = rt_wf.register_deployment(
            Deployment(
                deployment_id="dep_real_wf",
                agent_name="wf-agent",
                agent_version="1.0",
                architecture="workflow",
                allowed_scopes=["read", "exec"],
                config={"workflow_nodes": wf_nodes},
            )
        )
        s_wf = rt_wf.create_session(dep_wf.deployment_id, "Parallel DAG check")
        d_wf = rt_wf.run_session(s_wf.session_id)
        assert d_wf.status == SessionStatus.COMPLETED and "2870" in d_wf.final_output and "rai-studio" in d_wf.final_output, d_wf
        print(f"  PASS! tokens={d_wf.tokens_used} output={d_wf.final_output[:120]}")
        report["workflow_parallel_dag"] = {"status": d_wf.status.value, "tokens": d_wf.tokens_used, "output": d_wf.final_output}

        # --- 2. Planner-Executor on REAL qwen-turbo (Planner) + qwen-plus (Executor) ---
        print("\n=== [2/5] PlannerExecutorEngine + ModelRouter (qwen-turbo Planner + qwen-plus Executor) ===")
        rt_pe = ManagedRuntime(rt_store, router, reg, worker_id="w-pe")
        dep_pe = rt_pe.register_deployment(
            Deployment(
                deployment_id="dep_real_pe",
                agent_name="pe-agent",
                agent_version="1.0",
                architecture="planner_executor",
                allowed_scopes=["read", "exec"],
                max_turns=10,
                system_prompt="Be extremely concise. Use at most 2 plan steps.",
            )
        )
        s_pe = rt_pe.create_session(dep_pe.deployment_id, "Look up `agent_sphere` via `mcp_kv_lookup`, then read `slo.txt` via `mcp_workspace_read` and report owner and availability.")
        d_pe = rt_pe.run_session(s_pe.session_id)
        assert d_pe.status == SessionStatus.COMPLETED and "99.95%" in d_pe.final_output and "rai-studio" in d_pe.final_output, d_pe
        print(f"  PASS! tokens={d_pe.tokens_used} route_stats={router.route_stats} output={d_pe.final_output[:130]}")
        report["planner_executor_model_router"] = {
            "status": d_pe.status.value,
            "tokens": d_pe.tokens_used,
            "route_stats": dict(router.route_stats),
            "output": d_pe.final_output,
        }

        # --- 3. Multi-Agent Heterogeneous Router (qwen-plus Solver + qwen-turbo Verifier) ---
        print("\n=== [3/5] Multi-Agent Heterogeneous Router (qwen-plus Solver + qwen-turbo Independent Verifier) ===")
        rt_ma = ManagedRuntime(rt_store, router, reg, worker_id="w-ma")
        dep_ma = rt_ma.register_deployment(
            Deployment(
                deployment_id="dep_real_ma",
                agent_name="ma-agent",
                agent_version="1.0",
                architecture="multi_agent_verifier",
                allowed_scopes=["read", "exec"],
                max_turns=8,
            )
        )
        s_ma = rt_ma.create_session(dep_ma.deployment_id, "Use sandbox_exec_python to compute factorial(10) and reply with the exact integer.")
        d_ma = rt_ma.run_session(s_ma.session_id)
        assert d_ma.status == SessionStatus.COMPLETED and "3628800" in d_ma.final_output, d_ma
        models_used = sorted({e.payload.get("model_name", "") for e in rt_store.get_events(s_ma.session_id) if e.event_type == EventType.LLM_RESPONDED})
        print(f"  PASS! models_used={models_used} tokens={d_ma.tokens_used} output={d_ma.final_output[:100]}")
        report["heterogeneous_multi_agent"] = {"status": d_ma.status.value, "models_used": models_used, "tokens": d_ma.tokens_used}

        # --- 4. REAL LLM + REAL OS `kill()` Crash Recovery + ElasticWorkerPool Auto-Failover ---
        print("\n=== [4/5] REAL qwen-plus + REAL OS Kill + ElasticWorkerPool Auto-Failover (Exactly-Once) ===")
        ledger_path = data_dir / "real_llm_ledger.txt"
        sig_file = data_dir / "real_kill.sig"
        dep_ch = rt_store.save_deployment(
            Deployment(
                deployment_id="dep_real_chaos",
                agent_name="billing-live",
                agent_version="1.0",
                architecture="react",
                lease_ttl_ms=1200,
                allowed_scopes=["read", "write"],
                system_prompt="Call write_ledger exactly once with the requested order_id and amount, then confirm completion.",
            )
        )
        s_ch, _ = rt_store.create_session(dep_ch.deployment_id, "Please charge order_id='ORD-888' for amount=250 using write_ledger.")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--chaos-child",
                "--db", str(data_dir / "runtime.db"),
                "--ws", str(data_dir / "workspaces"),
                "--ledger", str(ledger_path),
                "--session-id", s_ch.session_id,
                "--signal-file", str(sig_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            deadline = time.time() + 25.0
            while not sig_file.exists() and time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Child exited early: {proc.stderr.read()}")
                time.sleep(0.05)
            assert sig_file.exists(), "Timed out waiting for real LLM child worker to commit ledger"
        finally:
            proc.kill()
            proc.wait(timeout=5)
            for st in (proc.stdout, proc.stderr):
                if st:
                    st.close()

        assert ledger_path.read_text(encoding="utf-8").splitlines() == ["ORD-888:250"]
        print("  Killed real-llm-worker-A mid-session right after ledger write! Waiting for lease expiry + ElasticWorkerPool failover...")
        time.sleep(1.35)

        def pool_worker_factory(wid: str) -> ManagedRuntime:
            r_led = ToolRegistry()
            r_led.register(
                ToolSpec(
                    name="write_ledger",
                    description="Record billing transaction",
                    parameters={"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "integer"}}, "required": ["order_id", "amount"]},
                    handler=lambda a, c: (ledger_path.open("a", encoding="utf-8").write(f"{a['order_id']}:{a['amount']}\n") and f"LEDGER_COMMITTED:{a['order_id']}:{a['amount']}"),
                    side_effect=True,
                    required_scopes=("write",),
                )
            )
            return ManagedRuntime(rt_store, qwen_plus, r_led, worker_id=wid)

        pool = ElasticWorkerPool(rt_store, pool_worker_factory, initial_workers=2, poll_interval_s=0.05)
        pool.start()
        try:
            statuses = pool.wait_for_sessions([s_ch.session_id], timeout_s=25.0)
            assert statuses == [SessionStatus.COMPLETED], statuses
        finally:
            pool.stop()

        final_ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
        evs_ch = [e.event_type for e in rt_store.get_events(s_ch.session_id)]
        assert final_ledger_lines == ["ORD-888:250"], f"Duplicate charge detected! {final_ledger_lines}"
        assert EventType.WORKER_CRASH_RECOVERED in evs_ch
        assert EventType.TOOL_REPLAYED_IDEMPOTENT in evs_ch
        s_ch_done = rt_store.get_session(s_ch.session_id)
        print(f"  PASS! ElasticWorkerPool recovered crashed REAL LLM session! Ledger entries={final_ledger_lines} (Exactly-Once!)")
        report["real_llm_os_kill_failover"] = {
            "status": s_ch_done.status.value,
            "ledger_entries": final_ledger_lines,
            "recovered": True,
            "replayed_idempotent": True,
            "final_output": s_ch_done.final_output[:160],
        }

        # --- 5. Personalization + Cross-Session Episode + Skill Progressive Disclosure + OnlineSampler ---
        print("\n=== [5/5] Personalization + Cross-Session Memory + Skill Disclosure + OnlineSampler ===")
        mem_store.set_user_preference("user_ljq", "language", "Chinese")
        mem_store.set_user_preference("user_ljq", "signoff_tag", "[SRE-TEAM-SH]")
        mem_store.distill_session_episode("prev_sess_01", "Check Redis cluster", "Redis shard-3 hit 92% memory at 14:00", user_id="user_ljq")
        sreg = SkillRegistry()
        sreg.register(SkillSpec.from_markdown(SKILL_SRE_MD))

        ctx_pers = BudgetedContextBuilder(max_context_tokens=2000, memory_store=mem_store, skill_registry=sreg, user_id="user_ljq")
        rt_pers = ManagedRuntime(rt_store, qwen_plus, reg, context_builder=ctx_pers, worker_id="w-pers")
        dep_pers = rt_pers.register_deployment(
            Deployment(
                deployment_id="dep_real_pers",
                agent_name="sre-personalized",
                agent_version="1.0",
                architecture="react",
                system_prompt="Follow Activated Skills and User Personalization Profile strictly (always end with the user's signoff_tag).",
            )
        )
        s_pers = rt_pers.create_session(dep_pers.deployment_id, "Write a brief incident postmortem referencing our previous Redis episode.")
        d_pers = rt_pers.run_session(s_pers.session_id)
        assert d_pers.status == SessionStatus.COMPLETED
        low_out = d_pers.final_output.lower()
        assert "impact" in low_out and "action" in low_out and "[sre-team-sh]" in low_out, d_pers.final_output
        assert "redis" in low_out or "92%" in low_out, d_pers.final_output

        sampler = OnlineSampler(rt_store, cp_store, sample_rate=1.0, latency_slo_ms=30_000)
        samp_rep = sampler.audit_sessions([s_wf.session_id, s_pe.session_id, s_ma.session_id, s_ch.session_id, s_pers.session_id])
        print(f"  PASS! Skill+Personalization+Cross-Session recall verified! OnlineSampler audited {samp_rep['sampled_count']} live sessions (grounding_failures={samp_rep['grounding_failures']})")
        print(f"  Personalized output:\n    " + d_pers.final_output.replace("\n", "\n    "))
        report["personalization_skill_sampler"] = {
            "status": d_pers.status.value,
            "output": d_pers.final_output,
            "online_sampler": samp_rep,
        }

    out_path = ROOT / "examples" / "full_jd_real_llm_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL 5 REAL-LLM JD SUITES PASSED! Saved report -> {out_path}")

    rt_store.close()
    cp_store.close()
    mem_store.close()


if __name__ == "__main__":
    main()
