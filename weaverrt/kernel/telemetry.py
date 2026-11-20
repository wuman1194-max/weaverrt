"""Hierarchical Trace Span recorder backed by RuntimeStore."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from weaverrt.domain import TraceSpan, new_id, now_ms
from weaverrt.runtime.store import RuntimeStore


class Tracer:
    def __init__(self, store: RuntimeStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._stack: list[str] = []

    @property
    def current_span_id(self) -> str:
        return self._stack[-1] if self._stack else ""

    @contextmanager
    def span(self, name: str, kind: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        sid = new_id("span")
        parent = self.current_span_id
        attrs: dict[str, Any] = dict(attributes or {})
        start = now_ms()
        status = "OK"
        self._stack.append(sid)
        try:
            yield attrs
        except Exception as exc:
            status = f"ERROR:{type(exc).__name__}"
            attrs["error"] = str(exc)
            raise
        finally:
            self._stack.pop()
            self.store.record_span(
                TraceSpan(
                    span_id=sid,
                    parent_span_id=parent,
                    session_id=self.session_id,
                    name=name,
                    kind=kind,
                    start_ms=start,
                    end_ms=now_ms(),
                    status=status,
                    attributes=attrs,
                )
            )
