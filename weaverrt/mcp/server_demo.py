"""Standalone MCP Server implementing JSON-RPC 2.0 over stdio (Model Context Protocol).

Exposes workspace-jailed tools dynamically discoverable via `tools/list` and callable via `tools/call`:
  - mcp_workspace_list : list files inside allowed root
  - mcp_workspace_read : read file inside allowed root (strict path-traversal guard)
  - mcp_kv_lookup      : lookup structured service config metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "weaverrt-mcp-demo", "version": "1.0.0"}

SERVICE_KB = {
    "agent_sphere": {"owner": "rai-studio", "slo_p95_ms": 2500, "sandbox": "linux-ns-v2"},
    "billing_svc": {"owner": "fin-infra", "idempotency": "required", "risk": "HIGH"},
    "search_rag": {"owner": "search-platform", "index": "fts5-hybrid", "top_k": 5},
}

TOOLS_MANIFEST = [
    {
        "name": "mcp_workspace_list",
        "description": "List files inside the sandboxed MCP workspace root.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "mcp_workspace_read",
        "description": "Read a UTF-8 file relative to the MCP workspace root.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "mcp_kv_lookup",
        "description": "Lookup internal service registry metadata by service name.",
        "inputSchema": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    },
]


def _safe_resolve(root: Path, rel: str) -> Path:
    root_res = root.resolve()
    target = (root_res / rel).resolve()
    if target != root_res and root_res not in target.parents:
        raise PermissionError(f"MCP path traversal blocked: {rel}")
    return target


def handle_tool_call(root: Path, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    try:
        if name == "mcp_workspace_list":
            root.mkdir(parents=True, exist_ok=True)
            entries = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
            return json.dumps({"files": entries}, ensure_ascii=False), False
        if name == "mcp_workspace_read":
            rel = str(args.get("path", ""))
            target = _safe_resolve(root, rel)
            if not target.is_file():
                return f"File not found: {rel}", True
            return target.read_text(encoding="utf-8", errors="replace"), False
        if name == "mcp_kv_lookup":
            svc = str(args.get("service", ""))
            if svc not in SERVICE_KB:
                return f"Service not found: {svc}", True
            return json.dumps(SERVICE_KB[svc], ensure_ascii=False), False
        return f"Unknown MCP tool: {name}", True
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", True


def serve_stdio(root: Path) -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        # Notifications have no id -> no response
        if req_id is None:
            continue

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                },
            }
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS_MANIFEST},
            }
        elif method == "tools/call":
            tname = str(params.get("name", ""))
            targs = dict(params.get("arguments") or {})
            text, is_err = handle_tool_call(root, tname, targs)
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": is_err,
                },
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Allowed workspace root directory")
    a = ap.parse_args()
    serve_stdio(Path(a.root))


if __name__ == "__main__":
    main()
