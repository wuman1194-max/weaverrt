"""MCP JSON-RPC 2.0 over stdio Client + Dynamic ToolRegistry Bridge."""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

from weaverrt.domain import RiskLevel
from weaverrt.kernel.tools import ToolContext, ToolRegistry, ToolSpec


class MCPStdioClient:
    """Connects to an external MCP Server subprocess via JSON-RPC 2.0 over stdin/stdout."""

    def __init__(self, command: list[str], *, env: dict[str, str] | None = None, timeout_s: float = 15.0) -> None:
        self.command = list(command)
        self.env = env
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self.server_info: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        if self._proc is not None:
            return self.server_info
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=self.env,
        )
        res = self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "weaverrt-mcp-client", "version": "1.0.0"},
            },
        )
        self.server_info = dict(res.get("serverInfo") or {})
        self._notify("notifications/initialized", {})
        return self.server_info

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
            finally:
                for stream in (proc.stdout, proc.stderr):
                    if stream:
                        try:
                            stream.close()
                        except Exception:
                            pass

    def __enter__(self) -> "MCPStdioClient":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                err_out = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"MCP server closed unexpectedly: {err_out}")
            resp = json.loads(line)
            if "error" in resp:
                raise RuntimeError(f"MCP error {resp['error'].get('code')}: {resp['error'].get('message')}")
            return dict(resp.get("result") or {})

    def list_tools(self) -> list[dict[str, Any]]:
        res = self._rpc("tools/list", {})
        return list(res.get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        res = self._rpc("tools/call", {"name": name, "arguments": arguments})
        is_err = bool(res.get("isError", False))
        texts = [str(c.get("text", "")) for c in (res.get("content") or []) if c.get("type") == "text"]
        return "\n".join(texts), is_err

    def register_into(
        self,
        registry: ToolRegistry,
        *,
        version: str | None = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        required_scopes: tuple[str, ...] = ("read",),
    ) -> list[str]:
        """Discover remote MCP tools and register them into WeaverRT's governed ToolRegistry."""
        ver = version or str(self.server_info.get("version") or "1.0.0")
        registered: list[str] = []
        for item in self.list_tools():
            tname = str(item["name"])
            desc = str(item.get("description") or f"MCP tool {tname}")
            schema = dict(item.get("inputSchema") or {"type": "object", "properties": {}})

            def make_handler(tool_name: str):
                def _handler(args: dict[str, Any], ctx: ToolContext) -> str:
                    out, is_err = self.call_tool(tool_name, args)
                    if is_err:
                        raise RuntimeError(out)
                    return out

                return _handler

            registry.register(
                ToolSpec(
                    name=tname,
                    version=ver,
                    description=f"[MCP:{self.server_info.get('name', 'remote')}] {desc}",
                    parameters=schema,
                    handler=make_handler(tname),
                    side_effect=False,
                    risk_level=risk_level,
                    required_scopes=required_scopes,
                )
            )
            registered.append(tname)
        return registered
