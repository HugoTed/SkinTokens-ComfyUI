"""TokenRig worker HTTP server (runs in isolated Python 3.11 venv)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import bottle
from bottle import request, response

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

os.chdir(PLUGIN_ROOT)
os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")


class _SafeStream:
    """stdout/stderr proxy that falls back to a log file when the console dies.

    On Windows, a worker whose parent console is gone (started hidden,
    detached, or from a terminal that has since closed) keeps invalid stdio
    handles: any print() or tqdm progress update then raises
    ``OSError: [Errno 22] Invalid argument``, failing whole /infer requests.
    The console can disappear at any time, so every write/flush is guarded.
    """

    _log_file = None

    def __init__(self, stream):
        self._stream = stream

    @classmethod
    def _log(cls):
        if cls._log_file is None:
            cls._log_file = open(
                PLUGIN_ROOT / "tokenrig-worker.log",
                "a",
                buffering=1,
                encoding="utf-8",
                errors="replace",
            )
        return cls._log_file

    def write(self, s):
        try:
            return self._stream.write(s)
        except (OSError, ValueError, AttributeError):
            try:
                return self._log().write(s)
            except OSError:
                return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError, AttributeError):
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except (OSError, ValueError, AttributeError):
            return False

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)

from config import get_default_model_ckpt, load_config, normalize_hf_path, normalize_output_path  # noqa: E402
from src.pipeline import (  # noqa: E402
    ensure_bpy_server,
    is_model_loaded,
    load_model,
    ping_bpy_server,
    resolve_output_path,
    restart_bpy_server,
    run_rig,
    start_bpy_server,
    wait_for_bpy_server,
)

_config = load_config()
WORKER_HOST = _config["worker"]["host"]
WORKER_PORT = _config["worker"]["port"]
_BPY_WATCHDOG_SECONDS = 5.0


def _looks_like_bpy_crash(exc: BaseException, tb: str) -> bool:
    text = f"{exc}\n{tb}".lower()
    needles = (
        "connection aborted",
        "connection reset",
        "connectionrefused",
        "remote end closed",
        "remotedisconnected",
        "10054",
        "bpy_server",
        "failed to establish a new connection",
        "max retries exceeded",
    )
    return any(n in text for n in needles)


def _recover_bpy_after_failure(exc: BaseException, tb: str) -> None:
    if not _looks_like_bpy_crash(exc, tb):
        return
    try:
        print("[TokenRig] infer failed with bpy connection error — recovering bpy_server")
        restart_bpy_server(python=sys.executable, cwd=PLUGIN_ROOT, timeout=120)
    except Exception:
        traceback.print_exc()


def _bpy_watchdog_loop() -> None:
    """Periodic heal: crashed bpy must not leave the worker degraded forever."""
    while True:
        time.sleep(_BPY_WATCHDOG_SECONDS)
        try:
            if ping_bpy_server(timeout=1.0):
                continue
            print("[TokenRig] watchdog: bpy_server down — restarting")
            restart_bpy_server(python=sys.executable, cwd=PLUGIN_ROOT, timeout=120)
        except Exception:
            traceback.print_exc()


def _read_json_body() -> dict:
    raw = request.body.read()  # type: ignore[attr-defined]
    if not raw:
        return {}
    return json.loads(raw)


def _json_response(payload: dict, status: int = 200):
    response.content_type = "application/json"  # type: ignore[attr-defined]
    response.status = status  # type: ignore[attr-defined]
    return json.dumps(payload)


def create_app() -> bottle.Bottle:
    app = bottle.Bottle()

    @app.route("/health", method="GET")  # type: ignore[misc]
    def health():
        bpy_ok = ping_bpy_server(timeout=1.0)
        return _json_response(
            {
                "status": "ok" if bpy_ok else "degraded",
                "bpy_ready": bpy_ok,
                "model_loaded": is_model_loaded(),
            }
        )

    @app.route("/restart_bpy", method="POST")  # type: ignore[misc]
    def restart_bpy_endpoint():
        """让编排端在发现 degraded 时主动拉起 bpy，避免干等。"""
        try:
            restart_bpy_server(python=sys.executable, cwd=PLUGIN_ROOT, timeout=120)
            ready = ping_bpy_server(timeout=1.0)
            return _json_response(
                {
                    "status": "ok" if ready else "degraded",
                    "bpy_ready": ready,
                    "model_loaded": is_model_loaded(),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb)
            return _json_response(
                {"status": "error", "error": str(exc), "traceback": tb},
                status=500,
            )

    @app.route("/load_model", method="POST")  # type: ignore[misc]
    def load_model_endpoint():
        try:
            data = _read_json_body()
            model_ckpt = data.get("model_ckpt")
            if not model_ckpt:
                model_ckpt = str(get_default_model_ckpt())
            hf_path = normalize_hf_path(data.get("hf_path"))
            message, ckpt = load_model(model_ckpt, hf_path=hf_path)
            return _json_response({"status": "ok", "message": message, "model_ckpt": ckpt})
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb)
            return _json_response({"status": "error", "error": str(exc), "traceback": tb}, status=500)

    @app.route("/infer", method="POST")  # type: ignore[misc]
    def infer_endpoint():
        try:
            # 坏模型可能已把 bpy 打挂；先恢复再接任务，避免堵死队列。
            ensure_bpy_server(python=sys.executable, cwd=PLUGIN_ROOT, timeout=120)

            data = _read_json_body()
            mesh_path = Path(data["mesh_path"]).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Mesh not found: {mesh_path}")

            export_format = data.get("export_format", "glb")
            output_path = normalize_output_path(data.get("output_path"))
            default_dir = PLUGIN_ROOT / "output" / "comfyui"
            default_dir.mkdir(parents=True, exist_ok=True)
            out_path = resolve_output_path(
                mesh_path,
                Path(output_path).resolve() if output_path else None,
                export_format=export_format,
                default_dir=default_dir,
            )

            model_ckpt = data.get("model_ckpt") or str(get_default_model_ckpt())
            hf_path = normalize_hf_path(data.get("hf_path"))

            results = run_rig(
                filepaths=[mesh_path],
                top_k=int(data.get("top_k", 5)),
                top_p=float(data.get("top_p", 0.95)),
                temperature=float(data.get("temperature", 1.0)),
                repetition_penalty=float(data.get("repetition_penalty", 2.0)),
                num_beams=int(data.get("num_beams", 10)),
                use_skeleton=bool(data.get("use_skeleton", False)),
                use_transfer=bool(data.get("use_transfer", False)),
                use_postprocess=bool(data.get("use_postprocess", False)),
                output_paths=[out_path],
                model_ckpt=model_ckpt,
                hf_path=hf_path,
            )
            return _json_response(
                {
                    "status": "ok",
                    "output_path": str(results[0]),
                    "mesh_path": str(mesh_path),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb)
            _recover_bpy_after_failure(exc, tb)
            return _json_response({"status": "error", "error": str(exc), "traceback": tb}, status=500)

    return app


def main():
    try:
        print(f"[TokenRig Worker] plugin root: {PLUGIN_ROOT}")
        start_bpy_server(python=sys.executable, cwd=PLUGIN_ROOT)
        wait_for_bpy_server(timeout=120)

        app = create_app()

        def run_server():
            bottle.run(app, host=WORKER_HOST, port=WORKER_PORT, server="tornado", quiet=False)

        threading.Thread(target=run_server, daemon=False).start()
        threading.Thread(
            target=_bpy_watchdog_loop,
            name="tokenrig-bpy-watchdog",
            daemon=True,
        ).start()
        print(f"[TokenRig Worker] listening on http://{WORKER_HOST}:{WORKER_PORT}")
        print("[TokenRig Worker] bpy watchdog enabled")

        threading.Event().wait()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
