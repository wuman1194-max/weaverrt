"""M1 Acceptance Tests: Harness Kernel decoupling, 3 Execution Architectures, Approval Suspend/Resume, Budget & Secret Redaction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from weaverrt.domain import (
    Deployment,
    EventType,
    Message,
    ModelResponse,
    RiskLevel,
    SessionStatus,
    TokenUsage,
    ToolCall,
)
from weaverrt.kernel.approval import PolicyEngine
from weaverrt.kernel.model import OpenAICompatibleAdapter, ScriptedModelAdapter
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime


def make_demo_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def write_note(args: dict, ctx: ToolContext) -> str:
        p = Path(ctx.workspace_path) / args["filename"]
        p.write_text(args["text"], encoding="utf-8")
        return f"wrote {len(args['text'])} bytes to {args['filename']}"

    def read_note(args: dict, ctx: ToolContext) -> str:
        p = Path(ctx.workspace_path) / args["filename"]
        return p.read_text(encoding="utf-8")

    def leak_secret(args: dict, ctx: ToolContext) -> str:
        return f"debug env token={ctx.secrets.get('DB_PASS', '')} ok"

    def prod_deploy(args: dict, ctx: ToolContext) -> str:
        p = Path(ctx.workspace_path) / "deployed.flag"
        p.write_text(args["release"], encoding="utf-8")
        return f"deployed {args['release']}"

    reg.register(
        ToolSpec(
            name="write_note",
            version="1.0.0",
            description="Write a text file into session workspace",
            parameters={
                "type": "object",
                "properties": {"filename": {"type": "string"}, "text": {"type": "string"}},
                "required": ["filename", "text"],
            },
            handler=write_note,
            side_effect=True,
            risk_level=RiskLevel.LOW,
            required_scopes=("write",),
        )
    )
    reg.register(
        ToolSpec(
            name="read_note",
            version="1.0.0",
            description="Read a text file from session workspace",
            parameters={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
            handler=read_note,
            side_effect=False,
            risk_level=RiskLevel.LOW,
            required_scopes=("read",),
        )
    )
    reg.register(
        ToolSpec(
            name="leak_secret",
            version="1.0.0",
            description="Simulates a leaky tool printing a secret",
            parameters={"type": "object", "properties": {}},
            handler=leak_secret,
            side_effect=False,
            risk_level=RiskLevel.LOW,
            required_scopes=("read",),
        )
    )
    reg.register(
        ToolSpec(
            name="prod_deploy",
            version="1.0.0",
            description="Deploy artifact to production (high risk)",
            parameters={
                "type": "object",
                "properties": {"release": {"type": "string"}},
                "required": ["release"],
            },
            handler=prod_deploy,
            side_effect=True,
            risk_level=RiskLevel.HIGH,
            required_scopes=("deploy",),
        )
    )
    return reg


class TestM1HarnessKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = RuntimeStore(self.root / "runtime.db", self.root / "workspaces")
        self.registry = make_demo_registry()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_01_react_architecture_and_traces(self) -> None:
        model = ScriptedModelAdapter(
            responses=[
                ModelResponse(
                    content="Let me write a note first.",
                    tool_calls=[ToolCall("c1", "write_note", {"filename": "out.txt", "text": "hello-react"})],
                ),
                ModelResponse(
                    content="Now read it back.",
                    tool_calls=[ToolCall("c2", "read_note", {"filename": "out.txt"})],
                ),
                ModelResponse(content="Verified file contents: hello-react"),
            ]
        )
        rt = ManagedRuntime(self.store, model, self.registry, worker_id="w-react")
        dep = rt.register_deployment(
            Deployment(deployment_id="dep_react", agent_name="demo", agent_version="1.0", architecture="react")
        )
        sess = rt.create_session(dep.deployment_id, "Create out.txt and verify it.")
        done = rt.run_session(sess.session_id)

        self.assertEqual(done.status, SessionStatus.COMPLETED)
        self.assertIn("hello-react", done.final_output)
        self.assertGreater(done.tokens_used, 0)
        self.assertEqual((Path(done.workspace_path) / "out.txt").read_text(encoding="utf-8"), "hello-react")

        traces = self.store.get_traces(sess.session_id)
        kinds = {t.kind for t in traces}
        self.assertTrue({"session", "turn", "context", "llm", "tool"}.issubset(kinds))

    def test_02_planner_executor_architecture(self) -> None:
        def handler(messages, tools, role_hint, idx):
            if role_hint == "planner":
                return ModelResponse(content=json.dumps({"steps": ["Write draft file", "Read draft file"]}))
            if role_hint == "executor":
                last_user = [m for m in messages if m.role == "user"][-1].content
                last_msg = messages[-1]
                if "Write draft file" in last_user and last_msg.role != "tool":
                    return ModelResponse(
                        tool_calls=[ToolCall("p1", "write_note", {"filename": "plan.txt", "text": "step1-ok"})]
                    )
                if "Read draft file" in last_user and last_msg.role != "tool":
                    return ModelResponse(tool_calls=[ToolCall("p2", "read_note", {"filename": "plan.txt"})])
                return ModelResponse(content=f"Step completed: {last_msg.content}")
            if role_hint == "synthesizer":
                return ModelResponse(content="FINAL PLAN RESULT: step1-ok")
            return ModelResponse(content="fallback")

        model = ScriptedModelAdapter(handler=handler)
        rt = ManagedRuntime(self.store, model, self.registry, worker_id="w-plan")
        dep = rt.register_deployment(
            Deployment(deployment_id="dep_plan", agent_name="planner", agent_version="1.0", architecture="planner_executor")
        )
        sess = rt.create_session(dep.deployment_id, "Write and verify plan.txt")
        done = rt.run_session(sess.session_id)

        self.assertEqual(done.status, SessionStatus.COMPLETED)
        self.assertEqual(done.final_output, "FINAL PLAN RESULT: step1-ok")
        events = [e.event_type for e in self.store.get_events(sess.session_id)]
        self.assertIn(EventType.PLAN_CREATED, events)
        self.assertEqual(events.count(EventType.PLAN_STEP_COMPLETED), 2)

    def test_03_multi_agent_solver_and_independent_verifier(self) -> None:
        solver_attempts = {"n": 0}
        verifier_saw_solver_history = {"leaked": False}

        def handler(messages, tools, role_hint, idx):
            if role_hint == "solver":
                solver_attempts["n"] += 1
                if solver_attempts["n"] == 1:
                    return ModelResponse(content="SECRET_SOLVER_THOUGHT: draft answer = 41")
                return ModelResponse(content="corrected answer = 42")
            if role_hint == "verifier":
                # Independent verifier must NOT see solver's private multi-turn history
                joined = "\n".join(m.content for m in messages)
                if "42" in joined:
                    return ModelResponse(content=json.dumps({"passed": True, "feedback": "exact match"}))
                # Round 1 check: only 2 messages (system + fresh verification prompt)
                if len(messages) != 2:
                    verifier_saw_solver_history["leaked"] = True
                return ModelResponse(content=json.dumps({"passed": False, "feedback": "Expected 42, got 41"}))
            return ModelResponse(content="unexpected")

        model = ScriptedModelAdapter(handler=handler)
        rt = ManagedRuntime(self.store, model, self.registry, worker_id="w-ma")
        dep = rt.register_deployment(
            Deployment(deployment_id="dep_ma", agent_name="solver", agent_version="1.0", architecture="multi_agent_verifier")
        )
        sess = rt.create_session(dep.deployment_id, "Compute 21 * 2")
        done = rt.run_session(sess.session_id)

        self.assertEqual(done.status, SessionStatus.COMPLETED)
        self.assertEqual(done.final_output, "corrected answer = 42")
        self.assertEqual(solver_attempts["n"], 2)
        self.assertFalse(verifier_saw_solver_history["leaked"])
        verdicts = [e.payload for e in self.store.get_events(sess.session_id) if e.event_type == EventType.VERIFIER_CHECKED]
        self.assertEqual(len(verdicts), 2)
        self.assertFalse(verdicts[0]["passed"])
        self.assertTrue(verdicts[1]["passed"])

    def test_04_high_risk_approval_suspend_and_resume_new_worker(self) -> None:
        llm_calls_w1 = ScriptedModelAdapter(
            responses=[
                ModelResponse(tool_calls=[ToolCall("d1", "prod_deploy", {"release": "v2.0.1"})]),
            ]
        )
        rt1 = ManagedRuntime(self.store, llm_calls_w1, self.registry, worker_id="worker-1")
        dep = rt1.register_deployment(
            Deployment(
                deployment_id="dep_appr",
                agent_name="deployer",
                agent_version="1.0",
                architecture="react",
                allowed_scopes=["read", "write", "deploy"],
            )
        )
        sess = rt1.create_session(dep.deployment_id, "Deploy v2.0.1 to prod")
        suspended = rt1.run_session(sess.session_id)
        self.assertEqual(suspended.status, SessionStatus.SUSPENDED_APPROVAL)
        self.assertFalse((Path(suspended.workspace_path) / "deployed.flag").exists())

        pending = self.store.list_approvals("PENDING")
        self.assertEqual(len(pending), 1)
        self.assertTrue(self.store.resolve_approval(pending[0]["approval_id"], approved=True, decided_by="lead_eng"))

        # Brand new Worker 2 resumes: Turn 0 LLM MUST NOT be re-called; only Turn 1 final response is called!
        llm_calls_w2 = ScriptedModelAdapter(responses=[ModelResponse(content="Deployment v2.0.1 confirmed!")])
        rt2 = ManagedRuntime(self.store, llm_calls_w2, self.registry, worker_id="worker-2")
        finished = rt2.run_session(sess.session_id)

        self.assertEqual(finished.status, SessionStatus.COMPLETED)
        self.assertEqual(finished.final_output, "Deployment v2.0.1 confirmed!")
        self.assertEqual(len(llm_calls_w2.calls), 1)  # Turn 0 was replayed from Event Log!
        self.assertEqual((Path(finished.workspace_path) / "deployed.flag").read_text(encoding="utf-8"), "v2.0.1")

    def test_05_secret_redaction_and_budget_enforcement(self) -> None:
        # 1. Secret redaction
        seen_tool_msg = {"text": ""}

        def handler(messages, tools, role_hint, idx):
            if idx == 0:
                return ModelResponse(tool_calls=[ToolCall("s1", "leak_secret", {})])
            seen_tool_msg["text"] = messages[-1].content
            return ModelResponse(content="done")

        rt = ManagedRuntime(
            self.store,
            ScriptedModelAdapter(handler=handler),
            self.registry,
            secrets={"DB_PASS": "super-secret-prod-999"},
        )
        dep = rt.register_deployment(Deployment(deployment_id="dep_sec", agent_name="sec", agent_version="1.0"))
        sess = rt.create_session(dep.deployment_id, "Run diagnostic")
        done = rt.run_session(sess.session_id)
        self.assertEqual(done.status, SessionStatus.COMPLETED)
        self.assertNotIn("super-secret-prod-999", seen_tool_msg["text"])
        self.assertIn("[REDACTED:DB_PASS]", seen_tool_msg["text"])

        # 2. Budget enforcement
        heavy_model = ScriptedModelAdapter(
            responses=[
                ModelResponse(
                    tool_calls=[ToolCall("b1", "write_note", {"filename": "a.txt", "text": "x"})],
                    usage=TokenUsage(prompt_tokens=600, completion_tokens=600, cost_usd=0.05),
                ),
                ModelResponse(content="should never reach here"),
            ]
        )
        rt_b = ManagedRuntime(self.store, heavy_model, self.registry)
        dep_b = rt_b.register_deployment(
            Deployment(deployment_id="dep_bud", agent_name="bud", agent_version="1.0", max_tokens=1000)
        )
        sess_b = rt_b.create_session(dep_b.deployment_id, "Spend too many tokens")
        exceeded = rt_b.run_session(sess_b.session_id)
        self.assertEqual(exceeded.status, SessionStatus.BUDGET_EXCEEDED)

    def test_06_openai_wire_payload_compatibility(self) -> None:
        adapter = OpenAICompatibleAdapter(model="deepseek-chat", api_key="test", base_url="https://api.deepseek.com/v1")
        payload = adapter.build_payload(
            [
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="", tool_calls=[ToolCall("c1", "read_note", {"filename": "a.txt"})]),
                Message(role="tool", content="abc", tool_call_id="c1", tool_name="read_note"),
            ],
            tools=self.registry.openai_schemas(),
        )
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["function"]["name"], "read_note")
        self.assertEqual(payload["messages"][3]["tool_call_id"], "c1")
        self.assertGreaterEqual(len(payload["tools"]), 4)


if __name__ == "__main__":
    unittest.main()
