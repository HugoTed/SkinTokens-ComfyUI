"""HTTP client for the TokenRig worker (ComfyUI-safe, no torch imports)."""

from __future__ import annotations

import atexit
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from bootstrap import ensure_worker_installed, venv_exists
from config import get_plugin_root, get_worker_python, get_worker_url, load_config
from process_util import popen_detached, terminate_process_tree

_worker_proc = None


def _worker_log_path() -> Path:
    return get_plugin_root() / "tokenrig-worker.log"


def _read_log_tail(path: Path, max_lines: int = 50) -> str:
    if not path.is_file():
        return (
            f"No log at {path}. Run manually:\n"
            f"  {get_worker_python() or '<venv>/bin/python'} worker/server.py"
        )
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-max_lines:])
        return f"Last lines from {path}:\n{tail}"
    except OSError as exc:
        return f"Could not read {path}: {exc}"


def _worker_exit_message(exit_code: Optional[int]) -> str:
    log_tail = _read_log_tail(_worker_log_path())
    code = "unknown" if exit_code is None else str(exit_code)
    return (
        f"TokenRig worker process exited before becoming ready (exit code {code}).\n"
        f"{log_tail}\n"
        "Tip: run `.tokenrig-venv/bin/python worker/server.py` in a terminal for full output."
    )


def ping_worker(timeout: float = 1.0) -> bool:
    try:
        resp = requests.get(f"{get_worker_url()}/health", timeout=timeout)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return data.get("bpy_ready", False)
    except Exception:
        return False


def _start_worker_process():
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return _worker_proc

    worker_python = get_worker_python()
    if worker_python is None:
        raise RuntimeError("TokenRig worker venv is not installed. Run bootstrap or TokenRig Setup node.")

    plugin_root = get_plugin_root()
    log_path = _worker_log_path()
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n--- worker start {datetime.now().isoformat()} ---\n")
    log_file.flush()
    proc = popen_detached(
        [str(worker_python), str(plugin_root / "worker" / "server.py")],
        cwd=str(plugin_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"[TokenRig Client] worker started (pid={proc.pid}), log: {log_path}")
    _worker_proc = proc

    def cleanup():
        print(f"[TokenRig Client] terminating worker (pid={proc.pid})")
        terminate_process_tree(proc)

    atexit.register(cleanup)
    return proc


def wait_for_worker(timeout: Optional[float] = None) -> None:
    config = load_config()
    timeout = timeout if timeout is not None else float(config["worker"]["startup_timeout"])
    t0 = time.time()
    while True:
        if ping_worker(timeout=2.0):
            print("[TokenRig Client] worker is ready")
            return
        proc = _worker_proc
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(_worker_exit_message(proc.returncode))
        if time.time() - t0 > timeout:
            raise RuntimeError(
                f"TokenRig worker failed to start within {timeout:.0f}s.\n"
                f"{_read_log_tail(_worker_log_path())}\n"
                "Tip: run `.tokenrig-venv/bin/python worker/server.py` in a terminal."
            )
        time.sleep(1.0)


def ensure_worker_running(auto_install: Optional[bool] = None) -> str:
    config = load_config()
    auto_install = config["worker"]["auto_install"] if auto_install is None else auto_install
    auto_start = config["worker"]["auto_start"]

    if ping_worker():
        return "TokenRig worker is already running."

    if auto_install and not venv_exists():
        ensure_worker_installed()

    if not venv_exists():
        raise RuntimeError(
            "TokenRig worker is not installed. Add TokenRig Setup node or run: python bootstrap.py"
        )

    if not auto_start:
        raise RuntimeError("TokenRig worker is not running and auto_start is disabled in config.json.")

    _start_worker_process()
    wait_for_worker()
    return "TokenRig worker started."


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    ensure_worker_running()
    url = f"{get_worker_url()}{path}"
    resp = requests.post(url, json=payload, timeout=3600)
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid worker response ({resp.status_code}): {resp.text}") from exc
    if resp.status_code >= 400 or data.get("status") == "error":
        raise RuntimeError(data.get("traceback") or data.get("error") or f"Worker error ({resp.status_code})")
    return data


def load_model(model_ckpt: str, hf_path: Optional[str] = None) -> tuple[str, str]:
    data = _post_json("/load_model", {"model_ckpt": model_ckpt, "hf_path": hf_path})
    return data.get("message", "Model loaded."), data.get("model_ckpt", model_ckpt)


def infer(
    mesh_path: str,
    output_path: Optional[str] = None,
    model_ckpt: Optional[str] = None,
    hf_path: Optional[str] = None,
    export_format: str = "glb",
    top_k: int = 5,
    top_p: float = 0.95,
    temperature: float = 1.0,
    repetition_penalty: float = 2.0,
    num_beams: int = 10,
    use_skeleton: bool = False,
    use_transfer: bool = False,
    use_postprocess: bool = False,
) -> str:
    payload = {
        "mesh_path": str(Path(mesh_path).resolve()),
        "output_path": output_path,
        "export_format": export_format,
        "model_ckpt": model_ckpt,
        "hf_path": hf_path,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
        "use_skeleton": use_skeleton,
        "use_transfer": use_transfer,
        "use_postprocess": use_postprocess,
    }
    data = _post_json("/infer", payload)
    return data["output_path"]
