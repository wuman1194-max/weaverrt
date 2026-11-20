"""SQLite WAL-backed Event Store, Lease Manager, Idempotency Cache, and Trace Store.

Design highlights (interview-ready):
  1. SQLite WAL mode + busy_timeout enables safe multi-process concurrent access
     (e.g., Worker A vs takeover Worker B vs Control Plane API reader).
  2. Append-only `events` table is the single source of truth for crash recovery:
     every turn transition, plan, LLM response, and tool result is committed atomically.
  3. Lease-based ownership (`worker_id`, `lease_expires_at_ms`, optimistic `version`)
     prevents split-brain execution while allowing dead workers to be replaced safely.
  4. `tool_idempotency` table guarantees Exactly-Once side-effect tool semantics even
     when a worker is SIGKILL'd after executing a tool but before recording the next turn.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from weaverrt.domain import (
    Deployment,
    Event,
    EventType,
    Session,
    SessionStatus,
    ToolResult,
    TraceSpan,
    Workspace,
    canonical_json,
    new_id,
    now_ms,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deployments (
    deployment_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    architecture TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    user_input TEXT NOT NULL,
    status TEXT NOT NULL,
    worker_id TEXT NOT NULL DEFAULT '',
    lease_expires_at_ms INTEGER NOT NULL DEFAULT 0,
    current_turn INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    final_output TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    FOREIGN KEY(deployment_id) REFERENCES deployments(deployment_id)
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    ended_at_ms INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq);

CREATE TABLE IF NOT EXISTS tool_idempotency (
    idempotency_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    executed_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idem_session ON tool_idempotency(session_id);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,  -- PENDING | APPROVED | REJECTED
    decided_by TEXT NOT NULL DEFAULT '',
    created_at_ms INTEGER NOT NULL,
    decided_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(session_id, status);

CREATE TABLE IF NOT EXISTS traces (
    span_id TEXT PRIMARY KEY,
    parent_span_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    attributes_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id, start_ms);
"""


class RuntimeStore:
    """Thread-safe and multi-process-safe SQLite WAL runtime store."""

    def __init__(self, db_path: str | Path, workspaces_root: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspaces_root = Path(workspaces_root or (self.db_path.parent / "workspaces"))
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._all_conns: list[sqlite3.Connection] = []
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level=None,  # explicit BEGIN IMMEDIATE for atomic multi-process safety
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute("PRAGMA foreign_keys=ON;")
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

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(SCHEMA_SQL)

    # ---------------- Deployments ----------------
    def save_deployment(self, dep: Deployment) -> Deployment:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO deployments
                (deployment_id, agent_name, agent_version, architecture, config_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dep.deployment_id,
                    dep.agent_name,
                    dep.agent_version,
                    dep.architecture,
                    canonical_json(dep.to_dict()),
                    dep.created_at_ms,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return dep

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        row = self._conn().execute(
            "SELECT config_json FROM deployments WHERE deployment_id = ?",
            (deployment_id,),
        ).fetchone()
        if row is None:
            return None
        return Deployment.from_dict(json.loads(row["config_json"]))

    # ---------------- Workspaces & Sessions ----------------
    def create_session(self, deployment_id: str, user_input: str, session_id: str | None = None) -> tuple[Session, Workspace]:
        sid = session_id or new_id("sess")
        ws_id = new_id("ws")
        ws_path = self.workspaces_root / sid
        ws_path.mkdir(parents=True, exist_ok=True)
        ts = now_ms()
        sess = Session(
            session_id=sid,
            deployment_id=deployment_id,
            workspace_path=str(ws_path),
            user_input=user_input,
            status=SessionStatus.PENDING,
            created_at_ms=ts,
            updated_at_ms=ts,
        )
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO sessions
                (session_id, deployment_id, workspace_path, user_input, status,
                 worker_id, lease_expires_at_ms, current_turn, tokens_used, cost_usd,
                 final_output, error, version, created_at_ms, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, '', 0, 0, 0, 0.0, '', '', 0, ?, ?)
                """,
                (sid, deployment_id, str(ws_path), user_input, sess.status.value, ts, ts),
            )
            conn.execute(
                """
                INSERT INTO events (session_id, turn_index, event_type, payload_json, created_at_ms)
                VALUES (?, 0, ?, ?, ?)
                """,
                (
                    sid,
                    EventType.SESSION_CREATED.value,
                    canonical_json({"deployment_id": deployment_id, "user_input": user_input, "workspace_path": str(ws_path)}),
                    ts,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return sess, Workspace(workspace_id=ws_id, session_id=sid, root_path=str(ws_path), created_at_ms=ts)

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            session_id=row["session_id"],
            deployment_id=row["deployment_id"],
            workspace_path=row["workspace_path"],
            user_input=row["user_input"],
            status=SessionStatus(row["status"]),
            worker_id=row["worker_id"],
            lease_expires_at_ms=row["lease_expires_at_ms"],
            current_turn=row["current_turn"],
            tokens_used=row["tokens_used"],
            cost_usd=row["cost_usd"],
            final_output=row["final_output"],
            error=row["error"],
            version=row["version"],
            created_at_ms=row["created_at_ms"],
            updated_at_ms=row["updated_at_ms"],
        )

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn().execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return self._row_to_session(row) if row is not None else None

    def list_sessions(self, limit: int = 50) -> list[Session]:
        rows = self._conn().execute(
            "SELECT * FROM sessions ORDER BY created_at_ms DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    # ---------------- Worker Lease & Takeover ----------------
    def acquire_lease(
        self,
        session_id: str,
        worker_id: str,
        ttl_ms: int,
        force_takeover: bool = False,
    ) -> tuple[bool, bool, str]:
        """Atomically claim or take over a Session lease.

        Returns (acquired, was_crash_recovery, previous_worker_id).
        """
        ts = now_ms()
        expires = ts + ttl_ms
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                return False, False, ""
            st = SessionStatus(row["status"])
            if st in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.BUDGET_EXCEEDED):
                conn.execute("ROLLBACK;")
                return False, False, row["worker_id"]

            prev_worker = row["worker_id"]
            prev_exp = row["lease_expires_at_ms"]
            lease_active = bool(prev_worker) and prev_worker != worker_id and prev_exp > ts
            if lease_active and not force_takeover:
                conn.execute("ROLLBACK;")
                return False, False, prev_worker

            recovered = bool(prev_worker) and prev_worker != worker_id and st == SessionStatus.RUNNING
            conn.execute(
                """
                UPDATE sessions
                SET status = ?, worker_id = ?, lease_expires_at_ms = ?,
                    version = version + 1, updated_at_ms = ?
                WHERE session_id = ? AND version = ?
                """,
                (SessionStatus.RUNNING.value, worker_id, expires, ts, session_id, row["version"]),
            )
            if conn.total_changes == 0:
                conn.execute("ROLLBACK;")
                return False, False, prev_worker

            if recovered:
                conn.execute(
                    """
                    INSERT INTO events (session_id, turn_index, event_type, payload_json, created_at_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        row["current_turn"],
                        EventType.WORKER_CRASH_RECOVERED.value,
                        canonical_json({"previous_worker": prev_worker, "new_worker": worker_id}),
                        ts,
                    ),
                )
            conn.execute(
                """
                INSERT INTO events (session_id, turn_index, event_type, payload_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    row["current_turn"],
                    EventType.WORKER_LEASE_ACQUIRED.value,
                    canonical_json({"worker_id": worker_id, "lease_expires_at_ms": expires}),
                    ts,
                ),
            )
            conn.execute("COMMIT;")
            return True, recovered, prev_worker
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def renew_lease(self, session_id: str, worker_id: str, ttl_ms: int) -> bool:
        ts = now_ms()
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cur = conn.execute(
                """
                UPDATE sessions
                SET lease_expires_at_ms = ?, updated_at_ms = ?
                WHERE session_id = ? AND worker_id = ? AND status = ?
                """,
                (ts + ttl_ms, ts, session_id, worker_id, SessionStatus.RUNNING.value),
            )
            conn.execute("COMMIT;")
            return cur.rowcount == 1
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def update_session_state(
        self,
        session_id: str,
        *,
        status: SessionStatus | None = None,
        current_turn: int | None = None,
        add_tokens: int = 0,
        add_cost_usd: float = 0.0,
        final_output: str | None = None,
        error: str | None = None,
        release_lease: bool = False,
    ) -> Session:
        ts = now_ms()
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown session: {session_id}")
            new_status = status.value if status is not None else row["status"]
            new_turn = current_turn if current_turn is not None else row["current_turn"]
            new_tokens = row["tokens_used"] + add_tokens
            new_cost = row["cost_usd"] + add_cost_usd
            new_output = final_output if final_output is not None else row["final_output"]
            new_error = error if error is not None else row["error"]
            new_worker = "" if release_lease else row["worker_id"]
            new_exp = 0 if release_lease else row["lease_expires_at_ms"]
            conn.execute(
                """
                UPDATE sessions
                SET status = ?, current_turn = ?, tokens_used = ?, cost_usd = ?,
                    final_output = ?, error = ?, worker_id = ?, lease_expires_at_ms = ?,
                    version = version + 1, updated_at_ms = ?
                WHERE session_id = ?
                """,
                (new_status, new_turn, new_tokens, new_cost, new_output, new_error, new_worker, new_exp, ts, session_id),
            )
            updated = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            conn.execute("COMMIT;")
            return self._row_to_session(updated)
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    # ---------------- Turns ----------------
    def upsert_turn(self, session_id: str, turn_index: int, status: str) -> str:
        ts = now_ms()
        turn_id = f"{session_id}_t{turn_index}"
        ended = ts if status in ("COMPLETED", "FAILED") else 0
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO turns (turn_id, session_id, turn_index, status, started_at_ms, ended_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, turn_index)
                DO UPDATE SET status = excluded.status, ended_at_ms = CASE WHEN excluded.ended_at_ms > 0 THEN excluded.ended_at_ms ELSE turns.ended_at_ms END
                """,
                (turn_id, session_id, turn_index, status, ts, ended),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return turn_id

    # ---------------- Append-only Event Log ----------------
    def append_event(
        self,
        session_id: str,
        turn_index: int,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> Event:
        ts = now_ms()
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cur = conn.execute(
                """
                INSERT INTO events (session_id, turn_index, event_type, payload_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, turn_index, event_type.value, canonical_json(payload), ts),
            )
            seq = int(cur.lastrowid)
            conn.execute("COMMIT;")
            return Event(
                seq=seq,
                session_id=session_id,
                turn_index=turn_index,
                event_type=event_type,
                payload=payload,
                created_at_ms=ts,
            )
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def get_events(self, session_id: str, after_seq: int = 0) -> list[Event]:
        rows = self._conn().execute(
            "SELECT * FROM events WHERE session_id = ? AND seq > ? ORDER BY seq ASC",
            (session_id, after_seq),
        ).fetchall()
        return [
            Event(
                seq=r["seq"],
                session_id=r["session_id"],
                turn_index=r["turn_index"],
                event_type=EventType(r["event_type"]),
                payload=json.loads(r["payload_json"]),
                created_at_ms=r["created_at_ms"],
            )
            for r in rows
        ]

    # ---------------- Exactly-Once Tool Idempotency Cache ----------------
    def get_idempotent_result(self, idempotency_key: str) -> ToolResult | None:
        row = self._conn().execute(
            "SELECT result_json FROM tool_idempotency WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        d = json.loads(row["result_json"])
        d["replayed"] = True
        return ToolResult.from_dict(d)

    def save_idempotent_result(
        self,
        idempotency_key: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> None:
        ts = now_ms()
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO tool_idempotency
                (idempotency_key, session_id, tool_name, args_json, status, result_json, executed_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    session_id,
                    tool_name,
                    canonical_json(arguments),
                    "OK" if result.ok else "ERROR",
                    canonical_json(result.to_dict()),
                    ts,
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    # ---------------- Approvals (Human-in-the-loop Gate) ----------------
    def create_approval(
        self,
        session_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
    ) -> str:
        aid = new_id("appr")
        ts = now_ms()
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO approvals
                (approval_id, session_id, turn_index, call_id, tool_name, args_json, reason, status, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (aid, session_id, turn_index, call_id, tool_name, canonical_json(arguments), reason, ts),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return aid

    def get_approval_for_call(self, session_id: str, call_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM approvals WHERE session_id = ? AND call_id = ? ORDER BY created_at_ms DESC LIMIT 1",
            (session_id, call_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def resolve_approval(self, approval_id: str, approved: bool, decided_by: str = "operator") -> bool:
        ts = now_ms()
        status = "APPROVED" if approved else "REJECTED"
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cur = conn.execute(
                """
                UPDATE approvals
                SET status = ?, decided_by = ?, decided_at_ms = ?
                WHERE approval_id = ? AND status = 'PENDING'
                """,
                (status, decided_by, ts, approval_id),
            )
            row = conn.execute("SELECT session_id, turn_index, call_id, tool_name FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
            if cur.rowcount == 1 and row is not None:
                conn.execute(
                    """
                    INSERT INTO events (session_id, turn_index, event_type, payload_json, created_at_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row["session_id"],
                        row["turn_index"],
                        EventType.APPROVAL_RESOLVED.value,
                        canonical_json({"approval_id": approval_id, "call_id": row["call_id"], "tool_name": row["tool_name"], "status": status, "decided_by": decided_by}),
                        ts,
                    ),
                )
            conn.execute("COMMIT;")
            return cur.rowcount == 1
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def list_approvals(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn().execute("SELECT * FROM approvals WHERE status = ? ORDER BY created_at_ms DESC", (status,)).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM approvals ORDER BY created_at_ms DESC").fetchall()
        return [dict(r) for r in rows]

    # ---------------- Telemetry Traces ----------------
    def record_span(self, span: TraceSpan) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                (span_id, parent_span_id, session_id, name, kind, start_ms, end_ms, status, attributes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.span_id,
                    span.parent_span_id,
                    span.session_id,
                    span.name,
                    span.kind,
                    span.start_ms,
                    span.end_ms,
                    span.status,
                    canonical_json(span.attributes),
                ),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def get_traces(self, session_id: str) -> list[TraceSpan]:
        rows = self._conn().execute(
            "SELECT * FROM traces WHERE session_id = ? ORDER BY start_ms ASC",
            (session_id,),
        ).fetchall()
        return [
            TraceSpan(
                span_id=r["span_id"],
                parent_span_id=r["parent_span_id"],
                session_id=r["session_id"],
                name=r["name"],
                kind=r["kind"],
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                status=r["status"],
                attributes=json.loads(r["attributes_json"]),
            )
            for r in rows
        ]
