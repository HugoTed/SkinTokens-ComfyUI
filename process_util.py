"""Cross-platform subprocess helpers (no heavy dependencies)."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any, Dict, List, Optional, Union


def popen_detached(
    args: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Start a child process in its own process group / session."""
    popen_kwargs: Dict[str, Any] = dict(
        args=args,
        cwd=cwd,
        env=env,
        stdout=None,
        stderr=None,
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
