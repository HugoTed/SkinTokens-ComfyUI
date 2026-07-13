"""Cross-platform subprocess helpers (no heavy dependencies)."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any, Dict, List, Optional, Set, Union


def popen_detached(
    args: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    stdout: Union[int, Any, None] = None,
    stderr: Union[int, Any, None] = None,
) -> subprocess.Popen:
    """Start a child process in its own process group / session."""
    popen_kwargs: Dict[str, Any] = dict(
        args=args,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Do not use preexec_fn=os.setsid here: it is unsafe when the parent
        # process has threads (e.g. ComfyUI).
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(**popen_kwargs)


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _pids_listening_on_port(port: int) -> Set[int]:
    """Best-effort: PIDs with a TCP listener on *port*."""
    pids: Set[int] = set()
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                text=True,
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except (OSError, subprocess.CalledProcessError):
            return pids
        needle = f":{port}"
        for line in out.splitlines():
            if "LISTENING" not in line.upper() or needle not in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local = parts[1]
            if not (local.endswith(needle) or local.endswith(f"]{needle}")):
                continue
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid > 0:
                pids.add(pid)
        return pids

    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
            text=True,
            errors="replace",
        )
        for tok in out.split():
            try:
                pids.add(int(tok))
            except ValueError:
                pass
    except (OSError, subprocess.CalledProcessError):
        try:
            out = subprocess.check_output(
                ["ss", "-ltnp", f"sport = :{port}"],
                text=True,
                errors="replace",
            )
            import re

            for m in re.finditer(r"pid=(\d+)", out):
                pids.add(int(m.group(1)))
        except (OSError, subprocess.CalledProcessError):
            pass
    return pids


def free_tcp_port(port: int) -> List[int]:
    """Kill processes listening on *port*. Returns killed PIDs."""
    killed: List[int] = []
    for pid in sorted(_pids_listening_on_port(port)):
        if pid == os.getpid():
            continue
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F", "/T"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
            else:
                os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (OSError, ProcessLookupError):
            pass
    return killed
