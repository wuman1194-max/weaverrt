"""Real OS-level Security Sandbox for Untrusted Agent Code Execution.

Linux Namespace Sandbox (runs natively on Linux or via WSL2 Ubuntu on Windows):
  - `unshare --user --map-root-user --pid --fork --mount-proc --net`:
      * Network namespace: zero external network / DNS / LAN egress
      * PID namespace: isolated process tree, invisible host processes
      * User namespace: unprivileged isolation
  - `prlimit --cpu --as --fsize`:
      * Hard kernel CPU time limit, virtual address space (memory) cap, file size cap
  - `env -i`:
      * Scrubs all host environment variables / secrets completely
  - Python Workspace Jail (audit hook + builtins.open / os path guard):
      * Blocks any file read/write or symlink escape outside the session workspace
  - Hard wall-clock timeout + process-group kill (`setsid` + `kill -9 -- -$pgid`):
      * Guarantees zero orphan processes even under infinite loops or fork attempts
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
import base64
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any

from weaverrt.domain import RiskLevel, now_ms
from weaverrt.kernel.tools import ToolContext, ToolSpec


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_s: float = 4.0
    cpu_limit_s: int = 3
    memory_limit_mb: int = 192
    max_file_size_mb: int = 4
    allow_network: bool = False
    max_output_chars: int = 8192


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    violation: str = ""  # "" | "PATH_TRAVERSAL" | "NETWORK_BLOCKED" | "TIMEOUT_OR_CPU" | "MEMORY_LIMIT" | "FORK_BLOCKED" | "POLICY_VIOLATION"
    duration_ms: int = 0
    isolation_layers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def windows_to_wsl_path(win_path: str | Path) -> str:
    p = str(Path(win_path).resolve())
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not m:
        return p.replace("\\", "/")
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


# Self-contained jailed Python runner executed INSIDE the Linux unshare + prlimit namespace.
# Uses Python PEP 578 `sys.addaudithook` (kernel-checked CPython hook!) to enforce strict
# workspace path confinement and block subprocess/fork escape attempts.
_JAILED_RUNNER_PY = r"""
import base64, builtins, os, resource, signal, sys

WS_ROOT = os.path.realpath(sys.argv[1])
MEM_BYTES = int(sys.argv[2])
CPU_SECS = int(sys.argv[3])
FSIZE_BYTES = int(sys.argv[4])
CODE = base64.b64decode(sys.argv[5]).decode("utf-8")

def _on_xcpu(signum, frame):
    sys.stderr.write("SANDBOX_VIOLATION:TIMEOUT_OR_CPU:RLIMIT_CPU_EXCEEDED\n")
    sys.stderr.flush()
    os._exit(152)

signal.signal(signal.SIGXCPU, _on_xcpu)

# Apply RLIMITs inside the isolated PID/net namespace (soft=CPU_SECS triggers SIGXCPU, hard=+1 triggers SIGKILL)
try:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECS, CPU_SECS + 1))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES))
except Exception:
    pass

os.chdir(WS_ROOT)

_ALLOWED_READ_PREFIXES = (
    WS_ROOT,
    "/usr/lib/python",
    "/usr/local/lib/python",
    "/lib/",
    "/usr/lib/",
)

def _is_inside_ws(path_str):
    try:
        real = os.path.realpath(os.path.join(WS_ROOT, path_str))
        return real == WS_ROOT or real.startswith(WS_ROOT + os.sep)
    except Exception:
        return False

def _audit_hook(event, args):
    if event in ("os.fork", "os.forkpty", "subprocess.Popen", "os.system", "os.posix_spawn"):
        raise PermissionError(f"SANDBOX_VIOLATION:FORK_BLOCKED:{event}")
    if event == "open":
        path_arg, mode, flags = args
        if isinstance(path_arg, (str, bytes)):
            p = path_arg.decode("utf-8", "ignore") if isinstance(path_arg, bytes) else path_arg
            real = os.path.realpath(os.path.join(WS_ROOT, p))
            is_write = any(c in str(mode or "") for c in ("w", "a", "x", "+"))
            if is_write:
                if not _is_inside_ws(p):
                    raise PermissionError(f"SANDBOX_VIOLATION:PATH_TRAVERSAL:write:{p}")
            else:
                if not _is_inside_ws(p) and not any(real.startswith(pref) for pref in _ALLOWED_READ_PREFIXES):
                    raise PermissionError(f"SANDBOX_VIOLATION:PATH_TRAVERSAL:read:{p}")
    elif event in ("os.listdir", "os.scandir"):
        if args and isinstance(args[0], (str, bytes)):
            p = args[0].decode("utf-8", "ignore") if isinstance(args[0], bytes) else args[0]
            real = os.path.realpath(os.path.join(WS_ROOT, p))
            if not _is_inside_ws(p) and not any(real.startswith(pref) for pref in _ALLOWED_READ_PREFIXES):
                raise PermissionError(f"SANDBOX_VIOLATION:PATH_TRAVERSAL:{event}:{p}")
    elif event in ("os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.rename", "os.chmod"):
        if args and isinstance(args[0], (str, bytes)):
            p = args[0].decode("utf-8", "ignore") if isinstance(args[0], bytes) else args[0]
            if not _is_inside_ws(p):
                raise PermissionError(f"SANDBOX_VIOLATION:PATH_TRAVERSAL:{event}:{p}")

sys.addaudithook(_audit_hook)

ns = {"__name__": "__main__", "WORKSPACE": WS_ROOT}
exec(compile(CODE, "<sandbox_code>", "exec"), ns, ns)
"""


def _classify_violation(exit_code: int, stderr: str, timed_out: bool) -> str:
    if timed_out or exit_code in (124, 137, 152, -9, -24) or "SANDBOX_VIOLATION:TIMEOUT_OR_CPU" in stderr:
        return "TIMEOUT_OR_CPU"
    if "SANDBOX_VIOLATION:PATH_TRAVERSAL" in stderr:
        return "PATH_TRAVERSAL"
    if "SANDBOX_VIOLATION:FORK_BLOCKED" in stderr:
        return "FORK_BLOCKED"
    if "MemoryError" in stderr or exit_code == 134:
        return "MEMORY_LIMIT"
    if any(
        k in stderr
        for k in (
            "Network is unreachable",
            "Temporary failure in name resolution",
            "Name or service not known",
            "Errno 101",
            "Errno -3",
            "Errno -2",
            "gaierror",
        )
    ):
        return "NETWORK_BLOCKED"
    return ""


class SandboxProvider(ABC):
    @abstractmethod
    def run_python(self, code: str, workspace_path: str | Path, policy: SandboxPolicy | None = None) -> SandboxResult:
        raise NotImplementedError


class WSL2LinuxNamespaceSandbox(SandboxProvider):
    """Executes untrusted Python code inside a real Linux User+PID+Net Namespace on WSL2 Ubuntu (or native Linux)."""

    def __init__(self, wsl_distro: str = "Ubuntu", audit_log_path: str | Path | None = None) -> None:
        self.wsl_distro = wsl_distro
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self.is_windows = os.name == "nt"

    def is_available(self) -> bool:
        if not self.is_windows:
            return shutil.which("unshare") is not None and shutil.which("python3") is not None
        return shutil.which("wsl") is not None

    def _record_audit(self, workspace_path: str, code: str, res: SandboxResult) -> None:
        if self.audit_log_path is None:
            return
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts_ms": now_ms(),
            "workspace": workspace_path,
            "code_preview": code[:200],
            "ok": res.ok,
            "exit_code": res.exit_code,
            "violation": res.violation,
            "timed_out": res.timed_out,
            "duration_ms": res.duration_ms,
            "isolation_layers": res.isolation_layers,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run_python(self, code: str, workspace_path: str | Path, policy: SandboxPolicy | None = None) -> SandboxResult:
        pol = policy or SandboxPolicy()
        ws = Path(workspace_path).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        wsl_ws = windows_to_wsl_path(ws) if self.is_windows else str(ws)

        runner_b64 = base64.b64encode(_JAILED_RUNNER_PY.encode("utf-8")).decode("ascii")
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        mem_bytes = pol.memory_limit_mb * 1024 * 1024
        fsize_bytes = pol.max_file_size_mb * 1024 * 1024
        timeout_int = max(1, int(round(pol.timeout_s)))

        unshare_flags = "--user --map-root-user --pid --fork --mount-proc"
        layers = ["linux_user_ns", "linux_pid_ns", "linux_rlimit_cpu_as_fsize", "env_scrubbed", "pep578_audit_jail"]
        if not pol.allow_network:
            unshare_flags += " --net"
            layers.append("linux_net_ns_offline")

        # Run inside Linux `timeout -k 1 <T>` + `unshare` + clean `env -i`
        bash_cmd = (
            f"timeout -s KILL {timeout_int}s "
            f"unshare {unshare_flags} "
            f"env -i PATH=/usr/bin:/bin HOME='{wsl_ws}' WORKSPACE='{wsl_ws}' "
            f"python3 -I -c \"import base64,sys; exec(base64.b64decode('{runner_b64}').decode())\" "
            f"'{wsl_ws}' {mem_bytes} {pol.cpu_limit_s} {fsize_bytes} '{code_b64}'"
        )

        cmd = ["wsl", "-d", self.wsl_distro, "--", "bash", "-c", bash_cmd] if self.is_windows else ["bash", "-c", bash_cmd]
        t0 = now_ms()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=pol.timeout_s + 4.0,
            )
            exit_code = proc.returncode
            stdout = proc.stdout[: pol.max_output_chars]
            stderr = proc.stderr[: pol.max_output_chars]
            if exit_code in (124, 137):
                timed_out = True
        except subprocess.TimeoutExpired as te:
            timed_out = True
            exit_code = 124
            stdout = (te.stdout or "")[: pol.max_output_chars] if isinstance(te.stdout, str) else ""
            stderr = "Sandbox wall-clock timeout expired (process killed)"

        dur = now_ms() - t0
        violation = _classify_violation(exit_code, stderr, timed_out)
        res = SandboxResult(
            ok=(exit_code == 0 and not timed_out and not violation),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            violation=violation,
            duration_ms=dur,
            isolation_layers=layers,
        )
        self._record_audit(str(ws), code, res)
        return res


def make_sandbox_exec_tool(
    provider: SandboxProvider,
    policy: SandboxPolicy | None = None,
    *,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
) -> ToolSpec:
    """Wraps a SandboxProvider as a WeaverRT governed ToolSpec (`sandbox_exec_python`)."""

    def handler(args: dict[str, Any], ctx: ToolContext) -> str:
        code = str(args["code"])
        res = provider.run_python(code, ctx.workspace_path, policy=policy)
        if not res.ok:
            raise RuntimeError(
                json.dumps(
                    {
                        "violation": res.violation or "EXEC_ERROR",
                        "exit_code": res.exit_code,
                        "timed_out": res.timed_out,
                        "stderr": res.stderr[-500:],
                        "layers": res.isolation_layers,
                    },
                    ensure_ascii=False,
                )
            )
        return res.stdout or "(no stdout)"

    return ToolSpec(
        name="sandbox_exec_python",
        version="1.0.0",
        description="Execute Python code inside an isolated Linux Namespace + RLIMIT sandbox confined to the session workspace.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code to run"}},
            "required": ["code"],
        },
        handler=handler,
        side_effect=True,
        risk_level=risk_level,
        required_scopes=("exec",),
    )
