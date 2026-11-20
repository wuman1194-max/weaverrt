"""M2 Chaos Acceptance Tests: Real OS Process Kill -> Lease Failover -> Exactly-Once Idempotent Recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from weaverrt.domain import Deployment, EventType, SessionStatus
from weaverrt.runtime.store import RuntimeStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_SCRIPT = PROJECT_ROOT / "tests" / "_chaos_worker_cli.py"


def wait_for_file(p: Path, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if p.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for chaos signal file: {p}")


class TestM2ChaosCrashRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "runtime.db"
        self.ws_root = self.root / "workspaces"
        self.ledger = self.root / "external_ledger.txt"
        self.store = RuntimeStore(self.db_path, self.ws_root)
        self.env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _spawn_worker(
        self,
        session_id: str,
        worker_id: str,
        scenario: str,
        pause_signal: Path | None = None,
        pause_at: str = "",
    ) -> subprocess.Popen:
        cmd = [
            sys.executable,
            str(CLI_SCRIPT),
            "--db", str(self.db_path),
            "--workspaces", str(self.ws_root),
            "--ledger", str(self.ledger),
            "--session-id", session_id,
            "--worker-id", worker_id,
            "--scenario", scenario,
        ]
        if pause_signal is not None:
            cmd += ["--pause-signal", str(pause_signal), "--pause-at", pause_at]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)

    def test_01_sigkill_after_side_effect_achieves_exactly_once_on_takeover(self) -> None:
        """Crash Window: side-effect executed & idempotency committed, process OS-killed BEFORE TOOL_SUCCEEDED & Turn 1.

        Verifies:
          1. Active lease prevents split-brain takeover before TTL expiry.
          2. After lease expires, brand new OS process worker-B takes over (`WORKER_CRASH_RECOVERED`).
          3. Side-effect tool hits idempotency cache (`TOOL_REPLAYED_IDEMPOTENT`) and NEVER executes twice.
          4. External ledger file has EXACTLY ONE line.
        """
        dep = self.store.save_deployment(
            Deployment(
                deployment_id="dep_chaos_react",
                agent_name="billing_agent",
                agent_version="1.0",
                architecture="react",
                lease_ttl_ms=600,
            )
        )
        sess, _ = self.store.create_session(dep.deployment_id, "Charge tx_001 for 100")
        sig_file = self.root / "kill_me.sig"

        # 1. Start real OS process Worker A; it executes charge_ledger, commits idempotency, then hangs
        proc_a = self._spawn_worker(
            sess.session_id, "worker-A", "react_idempotent", pause_signal=sig_file, pause_at="after_idempotency_commit"
        )
        try:
            wait_for_file(sig_file, timeout_s=10.0)
            # Verify split-brain protection BEFORE killing / lease expiry
            acquired, _, holder = self.store.acquire_lease(sess.session_id, "worker-intruder", ttl_ms=600)
            self.assertFalse(acquired)
            self.assertEqual(holder, "worker-A")
        finally:
            # 2. Hard OS-level kill (TerminateProcess / SIGKILL)
            proc_a.kill()
            proc_a.wait(timeout=5)
            for s in (proc_a.stdout, proc_a.stderr):
                if s:
                    s.close()

        # Ledger already has 1 physical write from crashed worker-A
        self.assertEqual(self.ledger.read_text(encoding="utf-8").splitlines(), ["tx_001:100"])

        # 3. Wait for worker-A's lease to expire, then launch real OS process Worker B
        time.sleep(0.75)
        proc_b = self._spawn_worker(sess.session_id, "worker-B", "react_idempotent")
        out_b, err_b = proc_b.communicate(timeout=15)
        self.assertEqual(proc_b.returncode, 0, msg=f"worker-B failed: {err_b}")
        res_b = json.loads(out_b.strip().splitlines()[-1])
        self.assertEqual(res_b["status"], SessionStatus.COMPLETED.value)
        self.assertEqual(res_b["final_output"], "COMPLETED_WITH:CHARGED:tx_001:100")

        # 4. EXACTLY-ONCE PROOF: external ledger STILL has exactly 1 line!
        self.assertEqual(self.ledger.read_text(encoding="utf-8").splitlines(), ["tx_001:100"])

        # 5. Event log proves crash recovery + idempotent replay + Turn 0 LLM was NOT re-called!
        events = self.store.get_events(sess.session_id)
        etypes = [e.event_type for e in events]
        self.assertIn(EventType.WORKER_CRASH_RECOVERED, etypes)
        self.assertIn(EventType.TOOL_REPLAYED_IDEMPOTENT, etypes)
        llm_turns = [e.turn_index for e in events if e.event_type == EventType.LLM_RESPONDED]
        self.assertEqual(llm_turns, [0, 1])  # Exactly 1 LLM call for Turn 0, 1 for Turn 1!

    def test_02_planner_executor_mid_plan_sigkill_resumes_at_exact_step(self) -> None:
        """Worker A creates 3-step plan, completes Step 0, then gets OS-killed before Step 1.
        Worker B takes over, skips Plan & Step 0, finishes Step 1 & Step 2 & Synthesizes!
        """
        dep = self.store.save_deployment(
            Deployment(
                deployment_id="dep_chaos_plan",
                agent_name="planner_agent",
                agent_version="1.0",
                architecture="planner_executor",
                lease_ttl_ms=500,
            )
        )
        sess, _ = self.store.create_session(dep.deployment_id, "Run 3 persistent steps")
        sig_file = self.root / "kill_plan.sig"

        proc_a = self._spawn_worker(
            sess.session_id, "worker-plan-A", "planner_midcrash", pause_signal=sig_file, pause_at="before_step_1"
        )
        try:
            wait_for_file(sig_file, timeout_s=10.0)
        finally:
            proc_a.kill()
            proc_a.wait(timeout=5)
            for s in (proc_a.stdout, proc_a.stderr):
                if s:
                    s.close()

        time.sleep(0.65)
        proc_b = self._spawn_worker(sess.session_id, "worker-plan-B", "planner_midcrash")
        out_b, err_b = proc_b.communicate(timeout=15)
        self.assertEqual(proc_b.returncode, 0, msg=f"worker-plan-B failed: {err_b}")
        res_b = json.loads(out_b.strip().splitlines()[-1])
        self.assertEqual(res_b["status"], SessionStatus.COMPLETED.value)
        self.assertEqual(res_b["final_output"], "ALL_3_STEPS_SYNTHESIZED")

        ws = Path(sess.workspace_path)
        # Step 0 MUST have run exactly once; Steps 1 and 2 each ran once after takeover
        for k in (0, 1, 2):
            lines = (ws / f"step_{k}.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines, [f"ran:{k}"], f"step_{k}.txt had unexpected executions: {lines}")

        events = self.store.get_events(sess.session_id)
        self.assertEqual(sum(1 for e in events if e.event_type == EventType.PLAN_CREATED), 1)
        self.assertIn(EventType.WORKER_CRASH_RECOVERED, [e.event_type for e in events])


if __name__ == "__main__":
    unittest.main()
