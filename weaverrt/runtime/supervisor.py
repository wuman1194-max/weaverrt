"""Elastic Worker Pool & Automated Failover Supervisor (Managed Runtime Control Loop).

Implements production Managed Runtime orchestration:
  - Dynamic worker scaling (`scale(target_workers)`)
  - Background Lease Heartbeat thread preventing false expiry during slow LLM/sandbox turns
  - Background Failover Reaper automatically discovering PENDING or expired-lease RUNNING sessions
    (crashed workers) and rescheduling them onto healthy standby workers
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from weaverrt.domain import SessionStatus, now_ms
from weaverrt.runtime.store import RuntimeStore
from weaverrt.runtime.worker import ManagedRuntime


class ElasticWorkerPool:
    def __init__(
        self,
        store: RuntimeStore,
        worker_factory: Callable[[str], ManagedRuntime],
        *,
        initial_workers: int = 2,
        poll_interval_s: float = 0.08,
    ) -> None:
        self.store = store
        self.worker_factory = worker_factory
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._workers: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._active_sessions: dict[str, str] = {}  # session_id -> worker_id
        self._next_idx = 1
        self.stats = {"completed": 0, "failed": 0, "reaped_failovers": 0}
        self._heartbeat_thread: threading.Thread | None = None
        self.scale(initial_workers)

    def start(self) -> None:
        if self._heartbeat_thread is None:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="lease-heartbeat", daemon=True)
            self._heartbeat_thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            entries = list(self._workers.values())
        for _, stop_ev in entries:
            stop_ev.set()
        for t, _ in entries:
            t.join(timeout=timeout_s)
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=timeout_s)
            self._heartbeat_thread = None

    def scale(self, target_workers: int) -> int:
        target = max(0, int(target_workers))
        with self._lock:
            # Scale up
            while len(self._workers) < target:
                wid = f"pool-worker-{self._next_idx}"
                self._next_idx += 1
                ev = threading.Event()
                t = threading.Thread(target=self._worker_loop, args=(wid, ev), name=wid, daemon=True)
                self._workers[wid] = (t, ev)
                t.start()
            # Scale down (graceful drain)
            while len(self._workers) > target:
                wid, (t, ev) = self._workers.popitem()
                ev.set()
            return len(self._workers)

    @property
    def worker_count(self) -> int:
        with self._lock:
            return len(self._workers)

    def _find_claimable_session(self) -> tuple[str | None, bool]:
        """Find a PENDING session OR an expired-lease RUNNING session (crashed worker failover)."""
        now = now_ms()
        for s in self.store.list_sessions(limit=200):
            if s.status == SessionStatus.PENDING:
                return s.session_id, False
            if s.status == SessionStatus.RUNNING and s.lease_expires_at_ms > 0 and s.lease_expires_at_ms < now:
                return s.session_id, True
        return None, False

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                active = list(self._active_sessions.items())
            for sid, wid in active:
                sess = self.store.get_session(sid)
                if sess is None:
                    continue
                dep = self.store.get_deployment(sess.deployment_id)
                if dep is not None and sess.status == SessionStatus.RUNNING:
                    self.store.renew_lease(sid, wid, dep.lease_ttl_ms)
            time.sleep(0.15)

    def _worker_loop(self, worker_id: str, local_stop: threading.Event) -> None:
        while not self._stop.is_set() and not local_stop.is_set():
            sid, was_expired = self._find_claimable_session()
            if sid is None:
                time.sleep(self.poll_interval_s)
                continue
            rt = self.worker_factory(worker_id)
            try:
                with self._lock:
                    self._active_sessions[sid] = worker_id
                done = rt.run_session(sid)
                with self._lock:
                    if was_expired:
                        self.stats["reaped_failovers"] += 1
                    if done.status == SessionStatus.COMPLETED:
                        self.stats["completed"] += 1
                    elif done.status in (SessionStatus.FAILED, SessionStatus.BUDGET_EXCEEDED):
                        self.stats["failed"] += 1
            except RuntimeError:
                # Another worker won the atomic lease race — totally normal under concurrency!
                pass
            finally:
                with self._lock:
                    self._active_sessions.pop(sid, None)

    def wait_for_sessions(self, session_ids: list[str], timeout_s: float = 15.0) -> list[SessionStatus]:
        terminal = {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.BUDGET_EXCEEDED,
            SessionStatus.SUSPENDED_APPROVAL,
        }
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            statuses = []
            for sid in session_ids:
                s = self.store.get_session(sid)
                statuses.append(s.status if s else SessionStatus.FAILED)
            if all(st in terminal for st in statuses):
                return statuses
            time.sleep(0.05)
        raise TimeoutError(f"Sessions did not finish within {timeout_s}s: {session_ids}")
