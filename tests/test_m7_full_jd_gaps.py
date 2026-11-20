"""M7 Full JD Coverage Tests:
  1. WorkflowEngine (DAG parallel fan-out/fan-in + conditional routing + crash recovery)
  2. ModelRouter (heterogeneous role routing + automatic fallback)
  3. ElasticWorkerPool (dynamic scaling + automatic expired-lease failover reaping)
  4. SkillRegistry (SKILL.md progressive disclosure)
  5. DelegationToken (scope attenuation, privilege escalation block, depth cap, tamper & TTL checks)
  6. Personalization Profile + Cross-Session Auto-Episode Distillation + GroundingVerifier
  7. Canary Rollout Weighted Routing + Real Cross-Process HTTP BYO Agent
  8. OnlineSampler (production sampling + fabricated citation detection + Bad Case feedback loop)
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from weaverrt.context.engine import BudgetedContextBuilder, verify_grounding
from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.control_plane.registry import (
    AgentArtifact,
    ArtifactStatus,
    ControlPlaneStore,
    HTTPBYOAgentAdapter,
)
from weaverrt.domain import (
    Deployment,
    EventType,
    ModelResponse,
    PolicyAction,
    RiskLevel,
    SessionStatus,
    ToolCall,
)
from weaverrt.eval.sampler import OnlineSampler
from weaverrt.kernel.approval import AgentIdentity, DelegationToken, PolicyEngine
from weaverrt.kernel.model import ModelRouter, ScriptedModelAdapter
from weaverrt.kernel.skills import SkillRegistry, SkillSpec
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec
from weaverrt.kernel.workflow import WorkflowNode
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.supervisor import ElasticWorkerPool
from weaverrt.runtime.worker import ManagedRuntime


SKILL_INCIDENT_MD = """---
name: incident-triage
version: 1.2.0
description: Structured incident triage runbook
triggers: [incident, oom, crash, latency]
allowed_tools: [fetch_metrics, read_artifact]
required_scopes: [read]
---
Always check P95 latency first, verify OOM killer logs, and output severity P0/P1/P2.
"""

SKILL_BILLING_MD = """---
name: billing-audit
version: 1.0.0
description: Financial ledger reconciliation
triggers: [invoice, ledger, refund]
allowed_tools: [query_ledger]
required_scopes: [read, billing]
---
Cross-check idempotency keys before issuing any refund.
"""


class TestM7FullJDCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rt_store = RuntimeStore(self.root / "runtime.db", self.root / "workspaces")
        self.cp_store = ControlPlaneStore(self.root / "cp.db")
        self.mem_store = LayeredMemoryStore(self.root / "memory.db")

    def tearDown(self) -> None:
        self.rt_store.close()
        self.cp_store.close()
        self.mem_store.close()
        self.tmp.cleanup()

    def test_01_workflow_dag_parallel_conditional_and_resume(self) -> None:
        call_counts = {"a": 0, "b": 0}
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="branch_a",
                description="Branch A",
                parameters={"type": "object", "properties": {}},
                handler=lambda a, c: (call_counts.__setitem__("a", call_counts["a"] + 1) or "A_DATA=10"),
                required_scopes=("read",),
            )
        )
        reg.register(
            ToolSpec(
                name="branch_b",
                description="Branch B",
                parameters={"type": "object", "properties": {}},
                handler=lambda a, c: (call_counts.__setitem__("b", call_counts["b"] + 1) or "B_DATA=20"),
                required_scopes=("read",),
            )
        )

        nodes = [
            WorkflowNode(node_id="fetch_a", kind="tool", tool_name="branch_a").to_dict(),
            WorkflowNode(node_id="fetch_b", kind="tool", tool_name="branch_b").to_dict(),
            WorkflowNode(
                node_id="optional_alert",
                kind="llm",
                depends_on=["fetch_a", "fetch_b"],
                condition={"node": "fetch_a", "contains": "ERROR"},  # False -> skipped!
                prompt_template="Alert on {fetch_a}",
            ).to_dict(),
            WorkflowNode(
                node_id="merge",
                kind="llm",
                depends_on=["fetch_a", "fetch_b", "optional_alert"],
                prompt_template="Combine {fetch_a} and {fetch_b} (alert={optional_alert})",
                role_hint="synthesizer",
            ).to_dict(),
        ]

        model = ScriptedModelAdapter(handler=lambda msgs, t, r, i: ModelResponse(content=f"MERGED:{msgs[-1].content}"))
        rt = ManagedRuntime(self.rt_store, model, reg, worker_id="w-wf-1")
        dep = rt.register_deployment(
            Deployment(
                deployment_id="dep_wf",
                agent_name="dag_agent",
                agent_version="1.0",
                architecture="workflow",
                config={"workflow_nodes": nodes},
            )
        )
        sess = rt.create_session(dep.deployment_id, "Run parallel DAG")
        # Pre-seed fetch_a as already checkpointed before crash -> verify resume skips branch_a!
        self.rt_store.append_event(
            sess.session_id,
            0,
            EventType.WORKFLOW_NODE_COMPLETED,
            {"node_id": "fetch_a", "kind": "tool", "output": "A_DATA=10"},
        )

        done = rt.run_session(sess.session_id)
        self.assertEqual(done.status, SessionStatus.COMPLETED)
        self.assertEqual(call_counts["a"], 0)  # Replayed from checkpoint!
        self.assertEqual(call_counts["b"], 1)  # Executed once!
        self.assertIn("A_DATA=10", done.final_output)
        self.assertIn("B_DATA=20", done.final_output)
        self.assertIn("[SKIPPED_BY_CONDITION]", done.final_output)

    def test_02_model_router_role_tiering_and_fallback(self) -> None:
        strong = ScriptedModelAdapter(responses=[ModelResponse(content="from-strong")], model_name="qwen-plus")
        fast = ScriptedModelAdapter(responses=[ModelResponse(content="from-fast")], model_name="qwen-turbo")
        failing = ScriptedModelAdapter(handler=lambda *a: (_ for _ in ()).throw(RuntimeError("503 overloaded")))
        router = ModelRouter(default_model=strong, routes={"verifier": fast, "flaky": failing}, fallback_model=fast)

        r1 = router.generate([], role_hint="solver")
        r2 = router.generate([], role_hint="verifier")
        r3 = router.generate([], role_hint="flaky")
        self.assertEqual(r1.model_name, "qwen-plus")
        self.assertEqual(r2.model_name, "qwen-turbo")
        self.assertEqual(r3.model_name, "qwen-turbo")
        self.assertEqual(router.route_stats["fallback"], 1)

    def test_03_elastic_worker_pool_scaling_and_auto_failover_reaping(self) -> None:
        reg = ToolRegistry()
        dep = self.rt_store.save_deployment(
            Deployment(deployment_id="dep_pool", agent_name="elastic", agent_version="1.0", lease_ttl_ms=250)
        )
        # Create 4 normal pending sessions + 1 crashed orphan session with expired lease!
        sids = [self.rt_store.create_session(dep.deployment_id, f"task-{i}")[0].session_id for i in range(4)]
        orphan = self.rt_store.create_session(dep.deployment_id, "orphan-task")[0]
        self.rt_store.acquire_lease(orphan.session_id, "dead-worker-99", ttl_ms=100)
        time.sleep(0.18)  # Dead worker's lease is now expired!
        sids.append(orphan.session_id)

        def factory(wid: str) -> ManagedRuntime:
            m = ScriptedModelAdapter(handler=lambda msgs, t, r, i: ModelResponse(content=f"handled by {wid}"))
            return ManagedRuntime(self.rt_store, m, reg, worker_id=wid)

        pool = ElasticWorkerPool(self.rt_store, factory, initial_workers=1, poll_interval_s=0.03)
        pool.start()
        try:
            self.assertEqual(pool.worker_count, 1)
            pool.scale(3)
            self.assertEqual(pool.worker_count, 3)
            statuses = pool.wait_for_sessions(sids, timeout_s=10.0)
            self.assertEqual(statuses, [SessionStatus.COMPLETED] * 5)
            self.assertGreaterEqual(pool.stats["reaped_failovers"], 1)
            evs = [e.event_type for e in self.rt_store.get_events(orphan.session_id)]
            self.assertIn(EventType.WORKER_CRASH_RECOVERED, evs)
        finally:
            pool.stop()

    def test_04_skill_registry_progressive_disclosure(self) -> None:
        sreg = SkillRegistry()
        sreg.register(SkillSpec.from_markdown(SKILL_INCIDENT_MD))
        sreg.register(SkillSpec.from_markdown(SKILL_BILLING_MD))

        builder = BudgetedContextBuilder(memory_store=self.mem_store, skill_registry=sreg)
        res = builder.build("sys", [], session_id="s1", workspace_path=str(self.root), query="Investigate P95 latency and OOM crash")
        self.assertEqual(res.stats["activated_skills"], ["incident-triage@1.2.0"])
        self.assertIn("Always check P95 latency first", res.messages[0].content)
        self.assertNotIn("issuing any refund", res.messages[0].content)

    def test_05_delegation_token_attenuation_depth_tamper_and_expiry(self) -> None:
        parent = AgentIdentity("coordinator", "1.0", "sess_del", allowed_scopes=("read", "write"))
        policy = PolicyEngine(signing_key="test-secret-key")

        # 1. Privilege escalation MUST raise PermissionError
        with self.assertRaises(PermissionError):
            DelegationToken.issue(parent, "sub_worker", ("read", "deploy"), signing_key="test-secret-key")

        # 2. Valid attenuated token (read only) works for read, denied for write!
        tok = DelegationToken.issue(parent, "sub_worker", ("read",), ttl_ms=5000, max_depth=1, signing_key="test-secret-key")
        child = AgentIdentity("sub_worker", "1.0", "sess_del", allowed_scopes=("read", "write"), delegation_token=tok)
        self.assertEqual(
            policy.evaluate(child, ToolCall("1", "t_read", {}), ("read",), RiskLevel.LOW).action,
            PolicyAction.ALLOW,
        )
        self.assertEqual(
            policy.evaluate(child, ToolCall("2", "t_write", {}), ("write",), RiskLevel.LOW).action,
            PolicyAction.DENY,
        )

        # 3. Depth cap prevents further re-delegation beyond max_depth=1
        with self.assertRaises(PermissionError):
            DelegationToken.issue(child, "grandchild", ("read",), signing_key="test-secret-key")

        # 4. Tampered signature rejected
        tampered = DelegationToken(
            issuer_agent=tok.issuer_agent,
            delegate_agent=tok.delegate_agent,
            session_id=tok.session_id,
            delegated_scopes=("read", "write"),  # forged!
            expires_at_ms=tok.expires_at_ms,
            depth=tok.depth,
            max_depth=tok.max_depth,
            signature=tok.signature,
        )
        bad_child = AgentIdentity("sub_worker", "1.0", "sess_del", allowed_scopes=("read", "write"), delegation_token=tampered)
        dec = policy.evaluate(bad_child, ToolCall("3", "t_read", {}), ("read",), RiskLevel.LOW)
        self.assertEqual(dec.action, PolicyAction.DENY)
        self.assertIn("signature invalid", dec.reason)

        # 5. Expired token rejected
        expired_tok = DelegationToken.issue(parent, "sub_exp", ("read",), ttl_ms=-10, signing_key="test-secret-key")
        exp_child = AgentIdentity("sub_exp", "1.0", "sess_del", allowed_scopes=("read",), delegation_token=expired_tok)
        dec_exp = policy.evaluate(exp_child, ToolCall("4", "t_read", {}), ("read",), RiskLevel.LOW)
        self.assertEqual(dec_exp.action, PolicyAction.DENY)
        self.assertIn("expired", dec_exp.reason)

    def test_06_personalization_auto_episode_and_grounding_verifier(self) -> None:
        self.mem_store.set_user_preference("user_alice", "output_style", "bullet_points_chinese")
        self.mem_store.set_user_preference("user_alice", "role", "infra_sre")
        self.mem_store.distill_session_episode("sess_prev", "Diagnose Redis OOM", "Raised maxmemory to 4GB", user_id="user_alice")

        builder = BudgetedContextBuilder(memory_store=self.mem_store, user_id="user_alice")
        built = builder.build("sys", [], session_id="sess_next", workspace_path=str(self.root), query="status")
        sys_txt = built.messages[0].content
        self.assertIn("bullet_points_chinese", sys_txt)
        self.assertIn("Raised maxmemory to 4GB", sys_txt)

        # Grounding verifier catches fabricated citation [KB:0x7f3a]
        ok1, bad1 = verify_grounding("Memory cap is 256MB [KB:policy_sandbox#c0].", ["[KB:policy_sandbox#c0]"])
        self.assertTrue(ok1)
        self.assertEqual(bad1, [])
        ok2, bad2 = verify_grounding("Limit is 128MB [KB:0x7f3a].", ["[KB:policy_sandbox#c0]"])
        self.assertFalse(ok2)
        self.assertEqual(bad2, ["[KB:0x7f3a]"])

    def test_07_canary_rollout_and_real_http_byo_agent(self) -> None:
        # Canary routing between v1 (RELEASED) and v2 (CANARY 20%)
        v1 = self.cp_store.register_artifact(AgentArtifact(name="feed-agent", version="1.0.0", owner="rec", last_gate_passed=True))
        v2 = self.cp_store.register_artifact(AgentArtifact(name="feed-agent", version="2.0.0", owner="rec", last_gate_passed=True))
        ok_r, _ = self.cp_store.promote_to_release("feed-agent", "1.0.0")
        self.assertTrue(ok_r)
        ok_c, _ = self.cp_store.promote_canary("feed-agent", "2.0.0", weight=0.25)
        self.assertTrue(ok_c)

        hits = {"1.0.0": 0, "2.0.0": 0}
        for i in range(400):
            chosen = self.cp_store.route_artifact("feed-agent", f"user_{i}")
            hits[chosen.version] += 1
        # Expect roughly ~25% canary (allow wide deterministic hash range 15%-35%)
        self.assertGreater(hits["2.0.0"], 60)
        self.assertLess(hits["2.0.0"], 140)
        self.assertEqual(self.cp_store.get_artifact("feed-agent", "2.0.0").status, ArtifactStatus.CANARY)

        # Real HTTP BYO Agent server
        class BYOHandler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or "0")
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                resp = json.dumps({"output": f"EXT_BYO_OK:{req['user_input']}", "tokens": 88}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), BYOHandler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            adapter = HTTPBYOAgentAdapter(f"http://127.0.0.1:{httpd.server_address[1]}")
            byo_art = AgentArtifact(name="ext-crew", version="1.0.0", owner="growth", runtime_type="byo", architecture="byo")
            out = adapter.invoke(byo_art, "sess_http_1", "rank posts", str(self.root))
            self.assertEqual(out["output"], "EXT_BYO_OK:rank posts")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_08_online_sampler_detects_hallucinated_citation_and_feeds_bad_cases(self) -> None:
        reg = ToolRegistry()
        dep = self.rt_store.save_deployment(Deployment(deployment_id="dep_prod", agent_name="prod-qa", agent_version="1.0"))
        # Session 1: hallucinates fake citation [KB:0x7f3a] (not in retrieved_citations!)
        rt_bad = ManagedRuntime(
            self.rt_store,
            ScriptedModelAdapter(responses=[ModelResponse(content="Memory cap is 128MB [KB:0x7f3a].")]),
            reg,
        )
        s_bad = rt_bad.create_session(dep.deployment_id, "What is sandbox memory cap?")
        rt_bad.run_session(s_bad.session_id)

        # Session 2: healthy clean session
        rt_ok = ManagedRuntime(
            self.rt_store,
            ScriptedModelAdapter(responses=[ModelResponse(content="All services healthy.")]),
            reg,
        )
        s_ok = rt_ok.create_session(dep.deployment_id, "Check health")
        rt_ok.run_session(s_ok.session_id)

        # OnlineSampler with sample_rate=0.0 still catches 100% of anomalies (hallucinated citation)!
        sampler = OnlineSampler(self.rt_store, self.cp_store, sample_rate=0.0, latency_slo_ms=10_000)
        report = sampler.audit_sessions([s_bad.session_id, s_ok.session_id])

        self.assertEqual(report["sampled_count"], 1)
        self.assertEqual(report["grounding_failures"], 1)
        self.assertEqual(len(report["bad_cases_created"]), 1)
        bcs = self.cp_store.list_bad_cases()
        self.assertEqual(bcs[0]["root_cause"], "hallucinated_citation")
        self.assertIn("[KB:0x7f3a]", bcs[0]["feedback"])


if __name__ == "__main__":
    unittest.main()
