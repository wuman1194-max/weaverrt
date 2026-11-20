"""Agent Control Plane Store: Declarative Agent Artifacts, Lifecycle State Machine, Gate Enforcement, Rollback & BYO Agent Governance."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Callable
import json
import sqlite3
import threading
import urllib.request
import yaml

from weaverrt.domain import Deployment, canonical_json, new_id, now_ms


class ArtifactStatus(str, Enum):
    DRAFT = "DRAFT"
    GATED = "GATED"
    CANARY = "CANARY"
    RELEASED = "RELEASED"
    ROLLED_BACK = "ROLLED_BACK"
    DEPRECATED = "DEPRECATED"


@dataclass
class EvalGateSpec:
    min_pass_rate: float = 0.85
    max_avg_tokens: int = 8000
    max_security_violations: int = 0
    max_p95_ms: int = 10000


@dataclass
class AgentArtifact:
    name: str
    version: str
    owner: str
    architecture: str = "react"  # "react" | "planner_executor" | "multi_agent_verifier" | "byo"
    runtime_type: str = "managed"  # "managed" | "byo"
    model: str = "scripted-deterministic-v1"
    system_prompt: str = "You are a governed production AI agent."
    allowed_scopes: list[str] = field(default_factory=lambda: ["read", "write", "compute"])
    max_turns: int = 10
    max_tokens: int = 16000
    max_cost_usd: float = 0.50
    context_strategy: str = "budgeted"  # "baseline" | "budgeted"
    max_context_tokens: int = 2000
    eval_gate: EvalGateSpec = field(default_factory=EvalGateSpec)
    status: ArtifactStatus = ArtifactStatus.DRAFT
    last_eval_run_id: str = ""
    last_gate_passed: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)
    released_at_ms: int = 0

    @property
    def artifact_id(self) -> str:
        return f"{self.name}@{self.version}"

    def to_deployment(self) -> Deployment:
        return Deployment(
            deployment_id=f"dep_{self.name}_{self.version}",
            agent_name=self.name,
            agent_version=self.version,
            architecture=self.architecture,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_cost_usd=self.max_cost_usd,
            allowed_scopes=list(self.allowed_scopes),
            system_prompt=self.system_prompt,
            config=dict(
                self.config,
                context_strategy=self.context_strategy,
                max_context_tokens=self.max_context_tokens,
                runtime_type=self.runtime_type,
                owner=self.owner,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["artifact_id"] = self.artifact_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentArtifact":
        gate_raw = d.get("eval_gate") or {}
        gate = gate_raw if isinstance(gate_raw, EvalGateSpec) else EvalGateSpec(**gate_raw)
        return cls(
            name=str(d["name"]),
            version=str(d["version"]),
            owner=str(d.get("owner", "unknown")),
            architecture=str(d.get("architecture", "react")),
            runtime_type=str(d.get("runtime_type", "managed")),
            model=str(d.get("model", "scripted-deterministic-v1")),
            system_prompt=str(d.get("system_prompt", "You are a governed production AI agent.")),
            allowed_scopes=list(d.get("allowed_scopes") or ["read", "write", "compute"]),
            max_turns=int(d.get("max_turns", 10)),
            max_tokens=int(d.get("max_tokens", 16000)),
            max_cost_usd=float(d.get("max_cost_usd", 0.50)),
            context_strategy=str(d.get("context_strategy", "budgeted")),
            max_context_tokens=int(d.get("max_context_tokens", 2000)),
            eval_gate=gate,
            status=ArtifactStatus(d.get("status", ArtifactStatus.DRAFT.value)),
            last_eval_run_id=str(d.get("last_eval_run_id", "")),
            last_gate_passed=bool(d.get("last_gate_passed", False)),
            config=dict(d.get("config") or {}),
            created_at_ms=int(d.get("created_at_ms") or now_ms()),
            released_at_ms=int(d.get("released_at_ms") or 0),
        )

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "AgentArtifact":
        return cls.from_dict(yaml.safe_load(yaml_text) or {})


CONTROL_PLANE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    owner TEXT NOT NULL,
    runtime_type TEXT NOT NULL,
    status TEXT NOT NULL,
    last_eval_run_id TEXT NOT NULL DEFAULT '',
    last_gate_passed INTEGER NOT NULL DEFAULT 0,
    spec_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    released_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_artifacts_name ON artifacts(name, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    avg_tokens REAL NOT NULL,
    total_cost_usd REAL NOT NULL,
    p50_ms INTEGER NOT NULL,
    p95_ms INTEGER NOT NULL,
    security_violations INTEGER NOT NULL,
    gate_passed INTEGER NOT NULL,
    gate_reasons_json TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bad_cases (
    bad_case_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    category TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    user_input TEXT NOT NULL,
    actual_output TEXT NOT NULL,
    feedback TEXT NOT NULL,
    exported_to_regression INTEGER NOT NULL DEFAULT 1,
    created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bad_cases_cat ON bad_cases(category, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS control_audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    target TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
"""


class ControlPlaneStore:
    """Unified governance store for Agent Artifacts, Release Gates, Eval Runs & Bad Case Feedback."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._all_conns: list[sqlite3.Connection] = []
        self._conn().executescript(CONTROL_PLANE_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            self._local.conn = conn
            with self._conns_lock:
                self._all_conns.append(conn)
        return conn

    def close(self) -> None:
        with self._conns_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()
        self._local.conn = None

    def _audit(self, conn: sqlite3.Connection, action: str, actor: str, target: str, detail: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO control_audit (action, actor, target, detail_json, created_at_ms) VALUES (?, ?, ?, ?, ?)",
            (action, actor, target, canonical_json(detail), now_ms()),
        )

    def register_artifact(self, art: AgentArtifact, actor: str = "developer") -> AgentArtifact:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts
                (artifact_id, name, version, owner, runtime_type, status, last_eval_run_id, last_gate_passed, spec_json, created_at_ms, released_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    art.artifact_id,
                    art.name,
                    art.version,
                    art.owner,
                    art.runtime_type,
                    art.status.value,
                    art.last_eval_run_id,
                    1 if art.last_gate_passed else 0,
                    canonical_json(art.to_dict()),
                    art.created_at_ms,
                    art.released_at_ms,
                ),
            )
            self._audit(conn, "REGISTER_ARTIFACT", actor, art.artifact_id, {"owner": art.owner, "runtime_type": art.runtime_type})
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return art

    def get_artifact(self, name: str, version: str) -> AgentArtifact | None:
        aid = f"{name}@{version}"
        row = self._conn().execute("SELECT spec_json FROM artifacts WHERE artifact_id = ?", (aid,)).fetchone()
        return AgentArtifact.from_dict(json.loads(row["spec_json"])) if row is not None else None

    def get_released(self, name: str) -> AgentArtifact | None:
        row = self._conn().execute(
            "SELECT spec_json FROM artifacts WHERE name = ? AND status = ? ORDER BY released_at_ms DESC LIMIT 1",
            (name, ArtifactStatus.RELEASED.value),
        ).fetchone()
        return AgentArtifact.from_dict(json.loads(row["spec_json"])) if row is not None else None

    def list_artifacts(self, name: str | None = None) -> list[AgentArtifact]:
        if name:
            rows = self._conn().execute(
                "SELECT spec_json FROM artifacts WHERE name = ? ORDER BY created_at_ms DESC", (name,)
            ).fetchall()
        else:
            rows = self._conn().execute("SELECT spec_json FROM artifacts ORDER BY created_at_ms DESC").fetchall()
        return [AgentArtifact.from_dict(json.loads(r["spec_json"])) for r in rows]

    def record_eval_run(self, summary: dict[str, Any]) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                (run_id, artifact_id, dataset_name, total_cases, passed_cases, pass_rate, avg_tokens,
                 total_cost_usd, p50_ms, p95_ms, security_violations, gate_passed, gate_reasons_json, details_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary["run_id"],
                    summary["artifact_id"],
                    summary["dataset_name"],
                    int(summary["total_cases"]),
                    int(summary["passed_cases"]),
                    float(summary["pass_rate"]),
                    float(summary["avg_tokens"]),
                    float(summary["total_cost_usd"]),
                    int(summary["p50_ms"]),
                    int(summary["p95_ms"]),
                    int(summary["security_violations"]),
                    1 if summary["gate_passed"] else 0,
                    canonical_json(summary["gate_reasons"]),
                    canonical_json(summary["case_results"]),
                    int(summary["created_at_ms"]),
                ),
            )
            # Update artifact's gate status
            row = conn.execute("SELECT spec_json FROM artifacts WHERE artifact_id = ?", (summary["artifact_id"],)).fetchone()
            if row is not None:
                art = AgentArtifact.from_dict(json.loads(row["spec_json"]))
                art.last_eval_run_id = summary["run_id"]
                art.last_gate_passed = bool(summary["gate_passed"])
                if art.last_gate_passed and art.status == ArtifactStatus.DRAFT:
                    art.status = ArtifactStatus.GATED
                conn.execute(
                    """
                    UPDATE artifacts
                    SET status = ?, last_eval_run_id = ?, last_gate_passed = ?, spec_json = ?
                    WHERE artifact_id = ?
                    """,
                    (
                        art.status.value,
                        art.last_eval_run_id,
                        1 if art.last_gate_passed else 0,
                        canonical_json(art.to_dict()),
                        art.artifact_id,
                    ),
                )
            self._audit(
                conn,
                "EVAL_RECORDED",
                "eval_runner",
                summary["artifact_id"],
                {"run_id": summary["run_id"], "pass_rate": summary["pass_rate"], "gate_passed": summary["gate_passed"]},
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def list_eval_runs(self, artifact_id: str | None = None) -> list[dict[str, Any]]:
        if artifact_id:
            rows = self._conn().execute(
                "SELECT * FROM eval_runs WHERE artifact_id = ? ORDER BY created_at_ms DESC", (artifact_id,)
            ).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM eval_runs ORDER BY created_at_ms DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["gate_passed"] = bool(d["gate_passed"])
            d["gate_reasons"] = json.loads(d.pop("gate_reasons_json"))
            d["case_results"] = json.loads(d.pop("details_json"))
            out.append(d)
        return out

    def promote_to_release(self, name: str, version: str, *, actor: str = "release_mgr") -> tuple[bool, str]:
        """Enforces Eval Gate before allowing an Agent Artifact into RELEASED state."""
        art = self.get_artifact(name, version)
        if art is None:
            return False, f"Artifact {name}@{version} not found"
        if not art.last_gate_passed:
            return (
                False,
                f"RELEASE BLOCKED: {art.artifact_id} has not passed Eval Gate (last_eval_run_id={art.last_eval_run_id or 'NONE'})",
            )
        ts = now_ms()
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            # Demote current RELEASED version of the same agent to ROLLED_BACK
            prev_rows = conn.execute(
                "SELECT artifact_id, spec_json FROM artifacts WHERE name = ? AND status = ?",
                (name, ArtifactStatus.RELEASED.value),
            ).fetchall()
            for pr in prev_rows:
                pa = AgentArtifact.from_dict(json.loads(pr["spec_json"]))
                pa.status = ArtifactStatus.ROLLED_BACK
                conn.execute(
                    "UPDATE artifacts SET status = ?, spec_json = ? WHERE artifact_id = ?",
                    (pa.status.value, canonical_json(pa.to_dict()), pa.artifact_id),
                )

            art.status = ArtifactStatus.RELEASED
            art.released_at_ms = ts
            conn.execute(
                "UPDATE artifacts SET status = ?, released_at_ms = ?, spec_json = ? WHERE artifact_id = ?",
                (art.status.value, ts, canonical_json(art.to_dict()), art.artifact_id),
            )
            self._audit(conn, "RELEASE_PROMOTED", actor, art.artifact_id, {"eval_run_id": art.last_eval_run_id})
            conn.execute("COMMIT;")
            return True, f"Released {art.artifact_id}"
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def rollback(self, name: str, target_version: str, *, actor: str = "oncall_sre") -> tuple[bool, str]:
        """One-click rollback to a previously gated/released version."""
        art = self.get_artifact(name, target_version)
        if art is None:
            return False, f"Target version {name}@{target_version} not found"
        if not art.last_gate_passed:
            return False, f"Cannot rollback to un-gated version {art.artifact_id}"
        ok, msg = self.promote_to_release(name, target_version, actor=f"rollback:{actor}")
        return ok, f"Rolled back -> {msg}" if ok else msg

    def promote_canary(self, name: str, version: str, weight: float = 0.2, *, actor: str = "release_mgr") -> tuple[bool, str]:
        """Promote a gated version to CANARY with traffic weight (0.0 < weight < 1.0)."""
        if not (0.0 < weight < 1.0):
            return False, "Canary weight must be strictly between 0 and 1"
        art = self.get_artifact(name, version)
        if art is None:
            return False, f"Artifact {name}@{version} not found"
        if not art.last_gate_passed:
            return False, f"CANARY BLOCKED: {art.artifact_id} has not passed Eval Gate"
        art.status = ArtifactStatus.CANARY
        art.config["canary_weight"] = round(float(weight), 4)
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                "UPDATE artifacts SET status = ?, spec_json = ? WHERE artifact_id = ?",
                (art.status.value, canonical_json(art.to_dict()), art.artifact_id),
            )
            self._audit(conn, "CANARY_PROMOTED", actor, art.artifact_id, {"weight": weight})
            conn.execute("COMMIT;")
            return True, f"Canary active for {art.artifact_id} at {weight:.0%} traffic"
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def route_artifact(self, name: str, routing_key: str) -> AgentArtifact | None:
        """Deterministic weighted traffic router between RELEASED stable and CANARY versions."""
        stable = self.get_released(name)
        canary_row = self._conn().execute(
            "SELECT spec_json FROM artifacts WHERE name = ? AND status = ? ORDER BY created_at_ms DESC LIMIT 1",
            (name, ArtifactStatus.CANARY.value),
        ).fetchone()
        if canary_row is None:
            return stable
        canary = AgentArtifact.from_dict(json.loads(canary_row["spec_json"]))
        if stable is None:
            return canary
        weight = float(canary.config.get("canary_weight", 0.1))
        bucket = int(hashlib.sha256(f"{name}:{routing_key}".encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
        return canary if bucket < weight else stable

    def save_bad_case(
        self,
        *,
        source_run_id: str,
        artifact_id: str,
        case_id: str,
        category: str,
        root_cause: str,
        user_input: str,
        actual_output: str,
        feedback: str,
    ) -> str:
        bcid = new_id("bc")
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO bad_cases
                (bad_case_id, source_run_id, artifact_id, case_id, category, root_cause, user_input, actual_output, feedback, exported_to_regression, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (bcid, source_run_id, artifact_id, case_id, category, root_cause, user_input, actual_output[:600], feedback[:600], now_ms()),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return bcid

    def list_bad_cases(self) -> list[dict[str, Any]]:
        rows = self._conn().execute("SELECT * FROM bad_cases ORDER BY created_at_ms DESC").fetchall()
        return [dict(r) for r in rows]


class BYOAgentRegistry:
    """Allows heterogeneous Bring-Your-Own (BYO) external Agents to be governed under WeaverRT Control Plane."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register(self, artifact_id: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._handlers[artifact_id] = handler

    def invoke(self, artifact: AgentArtifact, session_id: str, user_input: str, workspace_path: str) -> dict[str, Any]:
        fn = self._handlers.get(artifact.artifact_id)
        if fn is None:
            raise KeyError(f"BYO handler not registered for {artifact.artifact_id}")
        req = {
            "artifact_id": artifact.artifact_id,
            "owner": artifact.owner,
            "session_id": session_id,
            "user_input": user_input,
            "workspace_path": workspace_path,
            "max_tokens": artifact.max_tokens,
            "max_cost_usd": artifact.max_cost_usd,
        }
        return fn(req)


class HTTPBYOAgentAdapter:
    """Real cross-process HTTP Bring-Your-Own Agent adapter governed by Control Plane."""

    def __init__(self, endpoint_url: str, timeout_s: float = 15.0) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout_s = timeout_s

    def invoke(self, artifact: AgentArtifact, session_id: str, user_input: str, workspace_path: str) -> dict[str, Any]:
        payload = {
            "artifact_id": artifact.artifact_id,
            "owner": artifact.owner,
            "session_id": session_id,
            "user_input": user_input,
            "workspace_path": workspace_path,
            "max_tokens": artifact.max_tokens,
            "max_cost_usd": artifact.max_cost_usd,
        }
        req = urllib.request.Request(
            f"{self.endpoint_url}/invoke",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
