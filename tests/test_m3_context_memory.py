"""M3 Acceptance Tests: Token Budget Enforcement, 100KB Artifact Externalization, Compaction & Grounded FTS5 RAG."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from weaverrt.context.engine import BudgetedContextBuilder, make_read_artifact_tool
from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.domain import Deployment, Message, ModelResponse, RiskLevel, SessionStatus, ToolCall
from weaverrt.kernel.model import ScriptedModelAdapter
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime


class TestM3ContextAndMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = RuntimeStore(self.root / "runtime.db", self.root / "workspaces")
        self.mem = LayeredMemoryStore(self.root / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        self.mem.close()
        self.tmp.cleanup()

    def test_01_layered_memory_and_grounded_fts5_rag_bilingual(self) -> None:
        self.mem.set_working("sess_1", "target_env", "production-shanghai")
        self.mem.add_episode("sess_0", "Previous deployment rolled back due to memory limit 512MB.", tags=["incident"])
        self.mem.ingest_document(
            "runbook_sandbox",
            "安全沙箱规范：所有代码执行工具必须禁用公网连接并限制内存上限为 256MB。违规调用需触发人工审批门禁。",
            source="security_policy.md",
        )
        self.mem.ingest_document(
            "runbook_idempotency",
            "Idempotency standard: every side-effect tool call must compute SHA256(session_id, turn, call_id, args) before execution.",
            source="runtime_spec.md",
        )

        cn_hits = self.mem.search_kb("沙箱 内存上限 审批门禁", top_k=2)
        self.assertGreaterEqual(len(cn_hits), 1)
        self.assertEqual(cn_hits[0].doc_id, "runbook_sandbox")
        self.assertTrue(cn_hits[0].citation.startswith("[KB:runbook_sandbox#"))

        en_hits = self.mem.search_kb("SHA256 idempotency side-effect tool", top_k=2)
        self.assertGreaterEqual(len(en_hits), 1)
        self.assertEqual(en_hits[0].doc_id, "runbook_idempotency")

    def test_02_100kb_tool_output_externalization_and_30_turn_budget_stress(self) -> None:
        ws = self.root / "workspaces" / "sess_stress"
        ws.mkdir(parents=True, exist_ok=True)
        self.mem.set_working("sess_stress", "critical_key", "FACT-ALPHA-999")
        self.mem.ingest_document("kb_slo", "AgentSphere SLO target: P95 latency < 2.5s, crash recovery < 1s.", source="slo.md")

        # Construct 30 turns + one 100KB huge tool log output
        history: list[Message] = [Message(role="user", content="Analyze crash log and verify AgentSphere SLO.")]
        huge_log = "\n".join(f"2026-11-19 INFO line={i:04d} payload={'x' * 50}" for i in range(1, 1600))
        self.assertGreater(len(huge_log), 100_000)

        history.append(Message(role="assistant", tool_calls=[ToolCall("big_1", "dump_log", {})]))
        history.append(Message(role="tool", content=huge_log, tool_call_id="big_1", tool_name="dump_log"))
        for i in range(1, 30):
            history.append(Message(role="user", content=f"Follow-up turn {i}: check metric chunk {i} details."))
            history.append(Message(role="assistant", content=f"Checked chunk {i}, all nominal with code M-{i}."))

        builder = BudgetedContextBuilder(
            max_context_tokens=1500,
            max_tool_output_chars=500,
            keep_recent_messages=6,
            memory_store=self.mem,
        )
        res = builder.build(
            "You are an infra agent.",
            history,
            session_id="sess_stress",
            workspace_path=str(ws),
            query="AgentSphere SLO P95 latency",
        )

        # Must stay strictly under token budget and save >95% tokens vs raw!
        self.assertTrue(res.compacted)
        self.assertLessEqual(res.estimated_tokens, 1500)
        self.assertGreater(res.stats["token_saved_pct"], 90.0)
        self.assertEqual(res.externalized_artifacts, [".artifacts/tool_big_1.txt"])
        self.assertTrue((ws / ".artifacts" / "tool_big_1.txt").is_file())
        self.assertIn("[KB:kb_slo#c0]", res.retrieved_citations)

        # Pinned working memory fact + Grounded citation + initial user request MUST all survive!
        joined = "\n".join(m.content for m in res.messages)
        self.assertIn("FACT-ALPHA-999", joined)
        self.assertIn("[KB:kb_slo#c0]", joined)
        self.assertIn("Analyze crash log", joined)
        self.assertIn("tool_big_1.txt", joined)

    def test_03_end_to_end_agent_reads_externalized_artifact_on_demand(self) -> None:
        reg = ToolRegistry()

        def dump_big_file(args: dict, ctx: ToolContext) -> str:
            lines = [f"line {i}: noise" for i in range(1, 400)]
            lines[249] = "line 250: ROOT_CAUSE=OOM_KILLED_AT_256MB"
            return "\n".join(lines)

        reg.register(
            ToolSpec(
                name="dump_big_file",
                description="Return large diagnostic log",
                parameters={"type": "object", "properties": {}},
                handler=dump_big_file,
                required_scopes=("read",),
            )
        )
        reg.register(make_read_artifact_tool())

        def handler(messages, tools, role_hint, idx):
            if idx == 0:
                return ModelResponse(tool_calls=[ToolCall("log1", "dump_big_file", {})])
            if idx == 1:
                # Model sees externalized handle and selectively reads lines 248-252!
                self.assertIn("EXTERNALIZED ARTIFACT", messages[-1].content)
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            "rd1",
                            "read_artifact",
                            {"artifact_path": ".artifacts/tool_log1.txt", "start_line": 248, "end_line": 252},
                        )
                    ]
                )
            self.assertIn("ROOT_CAUSE=OOM_KILLED_AT_256MB", messages[-1].content)
            return ModelResponse(content="Found root cause: OOM_KILLED_AT_256MB")

        builder = BudgetedContextBuilder(max_context_tokens=1800, max_tool_output_chars=400, memory_store=self.mem)
        rt = ManagedRuntime(self.store, ScriptedModelAdapter(handler=handler), reg, context_builder=builder)
        dep = rt.register_deployment(Deployment(deployment_id="dep_m3", agent_name="diag", agent_version="1.0"))
        sess = rt.create_session(dep.deployment_id, "Find root cause in huge log")
        done = rt.run_session(sess.session_id)

        self.assertEqual(done.status, SessionStatus.COMPLETED)
        self.assertIn("OOM_KILLED_AT_256MB", done.final_output)


if __name__ == "__main__":
    unittest.main()
