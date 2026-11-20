"""M4 Acceptance Tests:
  1. Real MCP JSON-RPC 2.0 over stdio Server Discovery & Agent Invocation
  2. 6 Real Security Attacks against Real WSL2 Ubuntu Linux Namespace + RLIMIT + Audit-Hook Sandbox & RBAC Policy
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from weaverrt.domain import Deployment, EventType, ModelResponse, RiskLevel, SessionStatus, ToolCall
from weaverrt.kernel.model import ScriptedModelAdapter
from weaverrt.kernel.tools import ToolRegistry
from weaverrt.mcp.client import MCPStdioClient
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime
from weaverrt.sandbox.linux_sandbox import SandboxPolicy, WSL2LinuxNamespaceSandbox, make_sandbox_exec_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = PROJECT_ROOT / "weaverrt" / "mcp" / "server_demo.py"


class TestM4MCPAndSecuritySandbox(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = RuntimeStore(self.root / "runtime.db", self.root / "workspaces")
        self.audit_log = self.root / "sandbox_audit.jsonl"
        self.sandbox = WSL2LinuxNamespaceSandbox(audit_log_path=self.audit_log)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_01_real_mcp_jsonrpc_discovery_and_agent_execution(self) -> None:
        mcp_root = self.root / "mcp_data"
        mcp_root.mkdir()
        (mcp_root / "config.txt").write_text("MAX_WORKERS=16\nREGION=shanghai", encoding="utf-8")

        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
        with MCPStdioClient([sys.executable, str(MCP_SERVER_SCRIPT), "--root", str(mcp_root)], env=env) as client:
            self.assertEqual(client.server_info["name"], "weaverrt-mcp-demo")
            reg = ToolRegistry()
            names = client.register_into(reg, required_scopes=("read",))
            self.assertIn("mcp_kv_lookup", names)
            self.assertIn("mcp_workspace_read", names)

            # Verify MCP server itself blocks path traversal too!
            err_text, is_err = client.call_tool("mcp_workspace_read", {"path": "../runtime.db"})
            self.assertTrue(is_err)
            self.assertIn("path traversal blocked", err_text.lower())

            # Run a real WeaverRT Agent calling discovered MCP tools
            model = ScriptedModelAdapter(
                responses=[
                    ModelResponse(
                        tool_calls=[
                            ToolCall("m1", "mcp_kv_lookup", {"service": "agent_sphere"}),
                            ToolCall("m2", "mcp_workspace_read", {"path": "config.txt"}),
                        ]
                    ),
                    ModelResponse(content="MCP integration verified: rai-studio / shanghai"),
                ]
            )
            rt = ManagedRuntime(self.store, model, reg)
            dep = rt.register_deployment(Deployment(deployment_id="dep_mcp", agent_name="mcp_agent", agent_version="1.0"))
            sess = rt.create_session(dep.deployment_id, "Check agent_sphere metadata and config.txt")
            done = rt.run_session(sess.session_id)

            self.assertEqual(done.status, SessionStatus.COMPLETED)
            events = self.store.get_events(sess.session_id)
            tool_outs = [e.payload.get("output", "") for e in events if e.event_type == EventType.TOOL_SUCCEEDED]
            self.assertTrue(any("rai-studio" in o for o in tool_outs))
            self.assertTrue(any("REGION=shanghai" in o for o in tool_outs))

    def test_02_sandbox_benign_workspace_compute_succeeds(self) -> None:
        ws = self.root / "ws_ok"
        res = self.sandbox.run_python(
            "with open('result.txt', 'w') as f:\n    f.write(str(sum(range(101))))\nprint('SUM=' + open('result.txt').read())\n",
            ws,
            policy=SandboxPolicy(timeout_s=5.0, memory_limit_mb=256),
        )
        self.assertTrue(res.ok, msg=f"Benign run failed: {res}")
        self.assertIn("SUM=5050", res.stdout)
        self.assertEqual((ws / "result.txt").read_text(encoding="utf-8"), "5050")
        self.assertIn("linux_net_ns_offline", res.isolation_layers)
        self.assertIn("linux_pid_ns", res.isolation_layers)

    def test_03_six_real_security_attacks_all_blocked_and_audited(self) -> None:
        ws = self.root / "ws_attack"
        ws.mkdir()
        pol = SandboxPolicy(timeout_s=2.0, cpu_limit_s=1, memory_limit_mb=160, allow_network=False)

        # Attack 1: Path traversal trying to read /etc/passwd
        r1 = self.sandbox.run_python("print(open('/etc/passwd').read())", ws, pol)
        self.assertFalse(r1.ok)
        self.assertEqual(r1.violation, "PATH_TRAVERSAL")

        # Attack 2: CPU infinite loop -> killed by RLIMIT_CPU / timeout
        r2 = self.sandbox.run_python("while True:\n    pass", ws, pol)
        self.assertFalse(r2.ok)
        self.assertEqual(r2.violation, "TIMEOUT_OR_CPU")

        # Attack 3: Memory bomb (>160MB allocation)
        r3 = self.sandbox.run_python("x = bytearray(300 * 1024 * 1024)\nprint(len(x))", ws, pol)
        self.assertFalse(r3.ok)
        self.assertEqual(r3.violation, "MEMORY_LIMIT")

        # Attack 4: Network exfiltration inside --net isolated namespace
        r4 = self.sandbox.run_python(
            "import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=1)\nprint('LEAKED')", ws, pol
        )
        self.assertFalse(r4.ok)
        self.assertEqual(r4.violation, "NETWORK_BLOCKED")

        # Attack 5: Fork bomb / subprocess escape blocked by PEP 578 audit hook
        r5 = self.sandbox.run_python("import os\nos.fork()", ws, pol)
        self.assertFalse(r5.ok)
        self.assertEqual(r5.violation, "FORK_BLOCKED")

        # Attack 6a: Env secret stripping (`env -i`) inside sandbox
        old_val = os.environ.get("TOP_SECRET_HOST_KEY")
        os.environ["TOP_SECRET_HOST_KEY"] = "sk-host-secret-999999"
        try:
            r6 = self.sandbox.run_python("import os\nprint(dict(os.environ))", ws, pol)
            self.assertTrue(r6.ok, msg=f"env inspect failed: {r6}")
            self.assertNotIn("sk-host-secret-999999", r6.stdout)
            self.assertNotIn("TOP_SECRET_HOST_KEY", r6.stdout)
        finally:
            if old_val is None:
                os.environ.pop("TOP_SECRET_HOST_KEY", None)
            else:
                os.environ["TOP_SECRET_HOST_KEY"] = old_val

        # Attack 6b: Unauthorized scope escalation via PolicyEngine RBAC
        reg = ToolRegistry()
        reg.register(make_sandbox_exec_tool(self.sandbox, pol))
        model = ScriptedModelAdapter(
            responses=[
                ModelResponse(tool_calls=[ToolCall("x1", "sandbox_exec_python", {"code": "print(1)"})]),
                ModelResponse(content="Denied as expected"),
            ]
        )
        rt = ManagedRuntime(self.store, model, reg)
        # Deployment only has `read` scope, NOT `exec`!
        dep = rt.register_deployment(
            Deployment(deployment_id="dep_readonly", agent_name="reader", agent_version="1.0", allowed_scopes=["read"])
        )
        sess = rt.create_session(dep.deployment_id, "Try running code without exec scope")
        done = rt.run_session(sess.session_id)
        self.assertEqual(done.status, SessionStatus.COMPLETED)
        failed_tools = [e.payload for e in self.store.get_events(sess.session_id) if e.event_type == EventType.TOOL_FAILED]
        self.assertEqual(len(failed_tools), 1)
        self.assertIn("lacks required scopes", failed_tools[0]["error"])

        # Verify every sandbox invocation was written to the immutable JSONL security audit log!
        audit_lines = [json.loads(ln) for ln in self.audit_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertGreaterEqual(len(audit_lines), 6)
        recorded_violations = {a["violation"] for a in audit_lines}
        self.assertTrue({"PATH_TRAVERSAL", "TIMEOUT_OR_CPU", "MEMORY_LIMIT", "NETWORK_BLOCKED", "FORK_BLOCKED"}.issubset(recorded_violations))


if __name__ == "__main__":
    unittest.main()
