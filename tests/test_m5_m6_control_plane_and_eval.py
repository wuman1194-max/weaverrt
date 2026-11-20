"""M5 + M6 Acceptance Tests:
Full AgentSphere Production Lifecycle:
  Build (YAML Artifact) -> Evaluate (20-case benchmark) -> Gate (v1 BLOCKED)
  -> Harvest Bad Cases -> Evolve to v2 (Budgeted Context + Independent Verifier)
  -> Evaluate (v2 PASSED + Regression Diff) -> Release via Live HTTP Control Plane
  -> Observe (Traces/Events/Web Console) -> Rollback + BYO Agent Governance.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from weaverrt.context.engine import BudgetedContextBuilder, make_read_artifact_tool
from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.control_plane.registry import (
    AgentArtifact,
    ArtifactStatus,
    BYOAgentRegistry,
    ControlPlaneStore,
)
from weaverrt.control_plane.server import ControlPlaneServer
from weaverrt.domain import ModelResponse, RiskLevel, SessionStatus, ToolCall
from weaverrt.eval.engine import EvalCase, EvalRunner
from weaverrt.kernel.context import BaselineContextBuilder
from weaverrt.kernel.model import ScriptedModelAdapter
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime


ARTIFACT_V1_YAML = """
name: ops-agent
version: 1.0.0
owner: rai-infra-intern
architecture: react
runtime_type: managed
model: scripted-deterministic-v1
context_strategy: baseline
max_turns: 6
max_tokens: 1800
max_cost_usd: 0.50
allowed_scopes: [read, write, compute]
eval_gate:
  min_pass_rate: 0.90
  max_avg_tokens: 1500
  max_security_violations: 0
  max_p95_ms: 5000
"""

ARTIFACT_V2_YAML = """
name: ops-agent
version: 2.0.0
owner: rai-infra-intern
architecture: multi_agent_verifier
runtime_type: managed
model: scripted-deterministic-v1
context_strategy: budgeted
max_context_tokens: 900
max_turns: 8
max_tokens: 3000
max_cost_usd: 0.50
allowed_scopes: [read, write, compute]
eval_gate:
  min_pass_rate: 0.90
  max_avg_tokens: 1500
  max_security_violations: 0
  max_p95_ms: 5000
"""


def build_eval_dataset() -> list[EvalCase]:
    cases: list[EvalCase] = []
    # 12 standard tool_use cases (both v1 and v2 pass)
    for i in range(1, 13):
        cases.append(
            EvalCase(
                case_id=f"std_tool_{i:02d}",
                category="tool_use",
                user_input=f"Inspect service svc-{i} health status.",
                expected_substrings=[f"svc-{i}", "HEALTHY"],
                required_tools=["fetch_status"],
            )
        )
    # 4 long_context cases: tool returns 12KB log -> v1 (baseline) blows 1800 token budget; v2 externalizes & succeeds!
    for i in range(1, 5):
        cases.append(
            EvalCase(
                case_id=f"long_ctx_{i:02d}",
                category="long_context",
                user_input=f"Diagnose huge dump log #{i} and extract ERROR_CODE.",
                expected_substrings=[f"ERR_OOM_{i}"],
                required_tools=["fetch_huge_log"],
                max_tokens=1800,
            )
        )
    # 4 tricky reasoning cases: solver makes an off-by-one mistake on draft 1; v2's Independent Verifier catches & fixes it!
    for i in range(1, 5):
        cases.append(
            EvalCase(
                case_id=f"verify_edge_{i:02d}",
                category="reasoning",
                user_input=f"Verify exact quota calculation for tenant-{i}.",
                expected_substrings=[f"QUOTA_VERIFIED_{i * 100}"],
                forbidden_substrings=["UNVERIFIED_DRAFT"],
            )
        )
    return cases


def http_json(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class TestM5M6ControlPlaneAndEval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rt_store = RuntimeStore(self.root / "runtime.db", self.root / "workspaces")
        self.cp_store = ControlPlaneStore(self.root / "control_plane.db")
        self.mem_store = LayeredMemoryStore(self.root / "memory.db")
        self.mem_store.ingest_document("ops_kb", "Standard ops runbook: verify quota before releasing.", source="runbook.md")
        self.server = ControlPlaneServer(self.rt_store, self.cp_store, host="127.0.0.1", port=0)
        self.base_url = self.server.start()

    def tearDown(self) -> None:
        self.server.stop()
        self.rt_store.close()
        self.cp_store.close()
        self.mem_store.close()
        self.tmp.cleanup()

    def _make_runtime_factory(self):
        def factory(art: AgentArtifact, case: EvalCase) -> ManagedRuntime:
            reg = ToolRegistry()

            def fetch_status(args: dict, ctx: ToolContext) -> str:
                return f"service={args['service']} state=HEALTHY"

            def fetch_huge_log(args: dict, ctx: ToolContext) -> str:
                idx = args["idx"]
                lines = [f"2026-11-19 INFO noise line {k} " + ("z" * 40) for k in range(1, 220)]
                lines[150] = f"2026-11-19 FATAL ERROR_CODE=ERR_OOM_{idx}"
                return "\n".join(lines)

            reg.register(
                ToolSpec(
                    name="fetch_status",
                    description="Fetch service status",
                    parameters={"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
                    handler=fetch_status,
                    required_scopes=("read",),
                )
            )
            reg.register(
                ToolSpec(
                    name="fetch_huge_log",
                    description="Fetch 12KB service log dump",
                    parameters={"type": "object", "properties": {"idx": {"type": "integer"}}, "required": ["idx"]},
                    handler=fetch_huge_log,
                    required_scopes=("read",),
                )
            )
            reg.register(make_read_artifact_tool())

            # Case-driven deterministic model showing exact behavioral difference between v1 and v2
            def handler(messages, tools, role_hint, call_idx):
                if role_hint == "verifier":
                    cand = messages[-1].content
                    if "UNVERIFIED_DRAFT" in cand:
                        return ModelResponse(content=json.dumps({"passed": False, "feedback": "Remove UNVERIFIED_DRAFT and compute exact quota * 100"}))
                    return ModelResponse(content=json.dumps({"passed": True, "feedback": "Verified OK"}))

                if case.category == "tool_use":
                    svc = case.user_input.split("service ")[1].split()[0]
                    if not any(m.role == "tool" for m in messages):
                        return ModelResponse(tool_calls=[ToolCall("t1", "fetch_status", {"service": svc})])
                    return ModelResponse(content=f"Report: {svc} is HEALTHY.")

                if case.category == "long_context":
                    num = int(case.case_id.split("_")[-1])
                    tool_msgs = [m for m in messages if m.role == "tool"]
                    if not tool_msgs:
                        return ModelResponse(tool_calls=[ToolCall("h1", "fetch_huge_log", {"idx": num})])
                    if "EXTERNALIZED ARTIFACT" in tool_msgs[-1].content and len(tool_msgs) == 1:
                        return ModelResponse(
                            tool_calls=[
                                ToolCall(
                                    "r1",
                                    "read_artifact",
                                    {"artifact_path": ".artifacts/tool_h1.txt", "start_line": 149, "end_line": 153},
                                )
                            ]
                        )
                    return ModelResponse(content=f"Root cause identified: ERR_OOM_{num}")

                if case.category == "reasoning":
                    num = int(case.case_id.split("_")[-1])
                    has_feedback = any("Independent Verifier Feedback" in m.content for m in messages)
                    if not has_feedback:
                        return ModelResponse(content=f"UNVERIFIED_DRAFT quota={num * 99}")
                    return ModelResponse(content=f"Final verified quota: QUOTA_VERIFIED_{num * 100}")

                return ModelResponse(content="ok")

            ctx_builder = (
                BudgetedContextBuilder(
                    max_context_tokens=art.max_context_tokens,
                    max_tool_output_chars=400,
                    memory_store=self.mem_store,
                )
                if art.context_strategy == "budgeted"
                else BaselineContextBuilder()
            )
            return ManagedRuntime(self.rt_store, ScriptedModelAdapter(handler=handler), reg, context_builder=ctx_builder)

        return factory

    def test_01_full_build_eval_gate_release_rollback_and_badcase_evolution(self) -> None:
        dataset = build_eval_dataset()
        self.assertEqual(len(dataset), 20)

        # 1. BUILD: Register v1 and v2 from declarative YAML Artifacts
        v1 = self.cp_store.register_artifact(AgentArtifact.from_yaml(ARTIFACT_V1_YAML), actor="ci")
        v2 = self.cp_store.register_artifact(AgentArtifact.from_yaml(ARTIFACT_V2_YAML), actor="ci")

        # 2. GATE PROTECTION BEFORE EVAL: Releasing un-evaluated v1 via HTTP API must be BLOCKED (409)!
        code, body = http_json(f"{self.base_url}/api/artifacts/release", "POST", {"name": "ops-agent", "version": "1.0.0"})
        self.assertEqual(code, 409)
        self.assertFalse(body["ok"])
        self.assertIn("RELEASE BLOCKED", body["message"])

        # 3. EVALUATE v1 (Naive ReAct + Baseline Context) -> Fails long_context (budget) & reasoning (no verifier)
        judge = ScriptedModelAdapter(handler=lambda *a: ModelResponse(content="0.95"))
        runner = EvalRunner(self.rt_store, self.cp_store, judge_model=judge)
        run_v1 = runner.run_experiment(v1, "agentsphere_bench_v1", dataset, self._make_runtime_factory())

        self.assertFalse(run_v1["gate_passed"])
        self.assertEqual(run_v1["passed_cases"], 12)  # 12/20 = 60% < 90% gate!
        self.assertAlmostEqual(run_v1["pass_rate"], 0.60, places=2)

        # Release v1 MUST STILL BE BLOCKED by Gate!
        code, body = http_json(f"{self.base_url}/api/artifacts/release", "POST", {"name": "ops-agent", "version": "1.0.0"})
        self.assertEqual(code, 409)

        # 4. EVOLVE: Verify all 8 failed cases were harvested into Bad Cases with root-cause attribution!
        bad_cases = self.cp_store.list_bad_cases()
        self.assertEqual(len(bad_cases), 8)
        causes = {b["root_cause"] for b in bad_cases}
        self.assertIn("token_overrun", causes)
        self.assertIn("incorrect_answer", causes)

        # 5. EVALUATE v2 (Budgeted Context + Independent Verifier) -> Passes 20/20 (100%)!
        run_v2 = runner.run_experiment(v2, "agentsphere_bench_v1", dataset, self._make_runtime_factory())
        self.assertTrue(run_v2["gate_passed"], msg=f"v2 gate failed: {run_v2['gate_reasons']}")
        self.assertEqual(run_v2["passed_cases"], 20)
        self.assertEqual(run_v2["pass_rate"], 1.0)
        self.assertLess(run_v2["avg_tokens"], run_v1["avg_tokens"])

        # 6. REGRESSION DIFF: Zero regressions, all 8 Bad Cases fixed!
        diff = EvalRunner.compare_runs(run_v1, run_v2)
        self.assertTrue(diff["safe_to_promote"])
        self.assertEqual(len(diff["improved_cases"]), 8)
        self.assertEqual(len(diff["regressed_cases"]), 0)
        self.assertAlmostEqual(diff["pass_rate_delta"], 0.40, places=2)
        self.assertLess(diff["avg_tokens_delta_pct"], 0.0)

        # 7. RELEASE v2 via Live HTTP Control Plane API
        code, body = http_json(f"{self.base_url}/api/artifacts/release", "POST", {"name": "ops-agent", "version": "2.0.0"})
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(self.cp_store.get_released("ops-agent").version, "2.0.0")

        # 8. RELEASE v2.1 & ONE-CLICK ROLLBACK back to v2.0.0
        v21_spec = AgentArtifact.from_yaml(ARTIFACT_V2_YAML)
        v21_spec.version = "2.1.0"
        v21 = self.cp_store.register_artifact(v21_spec, actor="ci")
        run_v21 = runner.run_experiment(v21, "agentsphere_bench_v1", dataset[:5], self._make_runtime_factory())
        self.assertTrue(run_v21["gate_passed"])
        code, _ = http_json(f"{self.base_url}/api/artifacts/release", "POST", {"name": "ops-agent", "version": "2.1.0"})
        self.assertEqual(code, 200)
        self.assertEqual(self.cp_store.get_released("ops-agent").version, "2.1.0")

        code, rb = http_json(f"{self.base_url}/api/artifacts/rollback", "POST", {"name": "ops-agent", "version": "2.0.0"})
        self.assertEqual(code, 200)
        self.assertTrue(rb["ok"])
        self.assertEqual(self.cp_store.get_released("ops-agent").version, "2.0.0")
        self.assertEqual(self.cp_store.get_artifact("ops-agent", "2.1.0").status, ArtifactStatus.ROLLED_BACK)

        # 9. OBSERVE: Check live Control Plane endpoints + Web Dashboard HTML + Session Trace API
        code, evals_api = http_json(f"{self.base_url}/api/evals")
        self.assertEqual(code, 200)
        self.assertEqual(len(evals_api["eval_runs"]), 3)

        sample_sid = run_v2["case_results"][0]["session_id"]
        code, traces_api = http_json(f"{self.base_url}/api/sessions/{sample_sid}/traces")
        self.assertEqual(code, 200)
        self.assertGreaterEqual(len(traces_api["traces"]), 4)

        with urllib.request.urlopen(f"{self.base_url}/", timeout=5) as r:
            html = r.read().decode("utf-8")
            self.assertEqual(r.status, 200)
            self.assertIn("AgentSphere Control Plane", html)

    def test_02_byo_heterogeneous_agent_governance(self) -> None:
        byo_art = self.cp_store.register_artifact(
            AgentArtifact(name="legacy-langgraph-bot", version="1.0.0", owner="biz-team", architecture="byo", runtime_type="byo")
        )
        byo_reg = BYOAgentRegistry()
        byo_reg.register(
            byo_art.artifact_id,
            lambda req: {"output": f"BYO handled {req['user_input']} under owner={req['owner']}", "tokens": 120},
        )
        resp = byo_reg.invoke(byo_art, "sess_byo_1", "summarize feed", str(self.root))
        self.assertIn("owner=biz-team", resp["output"])


if __name__ == "__main__":
    unittest.main()
