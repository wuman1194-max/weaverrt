"""Layered Memory & Grounded Retrieval Store backed by SQLite + FTS5.

Three explicit memory tiers matching production Agent Infra design:
  1. Working Memory  — per-session pinned key-value state (facts, constraints, user prefs)
  2. Episodic Memory — cross-turn / cross-session interaction summaries
  3. Semantic KB     — chunked knowledge documents with SQLite FTS5 BM25 + CJK fallback & explicit citation IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import sqlite3
import threading
from typing import Any

from weaverrt.domain import canonical_json, new_id, now_ms


MEMORY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS working_memory (
    session_id TEXT NOT NULL,
    mem_key TEXT NOT NULL,
    mem_value TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id, mem_key)
);

CREATE TABLE IF NOT EXISTS episodic_memory (
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default',
    summary TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ep_user ON episodic_memory(user_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT NOT NULL,
    pref_key TEXT NOT NULL,
    pref_value TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (user_id, pref_key)
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    doc_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens_str TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    PRIMARY KEY (doc_id, chunk_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    source UNINDEXED,
    tokens_str
);
"""


def _tokenize_mixed(text: str) -> list[str]:
    """Tokenize mixed English + CJK text so SQLite FTS5 retrieves both seamlessly."""
    tokens: list[str] = []
    # English/number words
    for w in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        if len(w) >= 2:
            tokens.append(w)
    # CJK bigrams + unigrams for reliable Chinese retrieval without external tokenizer
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
        else:
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])
    return tokens


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source: str
    content: str
    score: float

    @property
    def citation(self) -> str:
        return f"[KB:{self.doc_id}#{self.chunk_id}]"


class LayeredMemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._all_conns: list[sqlite3.Connection] = []
        self._conn().executescript(MEMORY_SCHEMA_SQL)

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

    # ---------------- 1. Working Memory ----------------
    def set_working(self, session_id: str, key: str, value: str) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO working_memory (session_id, mem_key, mem_value, updated_at_ms)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, mem_key)
                DO UPDATE SET mem_value = excluded.mem_value, updated_at_ms = excluded.updated_at_ms
                """,
                (session_id, key, value, now_ms()),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def get_working(self, session_id: str) -> dict[str, str]:
        rows = self._conn().execute(
            "SELECT mem_key, mem_value FROM working_memory WHERE session_id = ? ORDER BY updated_at_ms ASC",
            (session_id,),
        ).fetchall()
        return {r["mem_key"]: r["mem_value"] for r in rows}

    # ---------------- 2. Episodic Memory ----------------
    def add_episode(self, session_id: str, summary: str, tags: list[str] | None = None, user_id: str = "default") -> str:
        eid = new_id("ep")
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO episodic_memory (episode_id, session_id, user_id, summary, tags_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (eid, session_id, user_id, summary, canonical_json(tags or []), now_ms()),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return eid

    def recent_episodes(self, user_id: str = "default", limit: int = 5) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM episodic_memory WHERE user_id = ? ORDER BY created_at_ms DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "episode_id": r["episode_id"],
                "session_id": r["session_id"],
                "summary": r["summary"],
                "tags": json.loads(r["tags_json"]),
                "created_at_ms": r["created_at_ms"],
            }
            for r in rows
        ]

    # ---------------- 2b. Personalization / User Profile Memory ----------------
    def set_user_preference(self, user_id: str, key: str, value: str) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, pref_key, pref_value, updated_at_ms)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, pref_key)
                DO UPDATE SET pref_value = excluded.pref_value, updated_at_ms = excluded.updated_at_ms
                """,
                (user_id, key, value, now_ms()),
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def get_user_profile(self, user_id: str) -> dict[str, str]:
        rows = self._conn().execute(
            "SELECT pref_key, pref_value FROM user_profiles WHERE user_id = ? ORDER BY pref_key ASC",
            (user_id,),
        ).fetchall()
        return {r["pref_key"]: r["pref_value"] for r in rows}

    def distill_session_episode(self, session_id: str, user_input: str, final_output: str, *, user_id: str = "default") -> str:
        """Automatically distill a finished session into cross-session Episodic Memory."""
        summary = f"Task='{user_input[:100]}' -> Result='{final_output[:160]}'"
        return self.add_episode(session_id, summary, tags=["auto_distilled"], user_id=user_id)

    # ---------------- 3. Semantic Knowledge Base (Grounded RAG) ----------------
    def ingest_document(
        self,
        doc_id: str,
        text: str,
        *,
        source: str = "internal_kb",
        chunk_size: int = 360,
        overlap: int = 60,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        chunks: list[str] = []
        clean = text.strip()
        if not clean:
            return []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(clean), step):
            piece = clean[i : i + chunk_size].strip()
            if piece:
                chunks.append(piece)
            if i + chunk_size >= len(clean):
                break

        ts = now_ms()
        ids: list[str] = []
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            # Replace prior chunks for same doc_id
            conn.execute("DELETE FROM kb_fts WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))

            for idx, piece in enumerate(chunks):
                cid = f"c{idx}"
                toks = " ".join(_tokenize_mixed(piece))
                conn.execute(
                    """
                    INSERT INTO kb_chunks (chunk_id, doc_id, source, content, tokens_str, metadata_json, created_at_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cid, doc_id, source, piece, toks, canonical_json(metadata or {}), ts),
                )
                conn.execute(
                    "INSERT INTO kb_fts (chunk_id, doc_id, source, tokens_str) VALUES (?, ?, ?, ?)",
                    (cid, doc_id, source, toks),
                )
                ids.append(cid)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        return ids

    def search_kb(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        qtoks = list(dict.fromkeys(_tokenize_mixed(query)))
        if not qtoks:
            return []
        # FTS5 OR query with BM25 ranking
        match_expr = " OR ".join(f'"{t}"' for t in qtoks[:20])
        rows = self._conn().execute(
            """
            SELECT c.chunk_id, c.doc_id, c.source, c.content, bm25(kb_fts) AS rank
            FROM kb_fts
            JOIN kb_chunks c ON c.chunk_id = kb_fts.chunk_id AND c.doc_id = kb_fts.doc_id
            WHERE kb_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (match_expr, top_k),
        ).fetchall()
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                source=r["source"],
                content=r["content"],
                score=round(-float(r["rank"]), 4),
            )
            for r in rows
        ]
