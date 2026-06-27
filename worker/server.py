"""TokenRig worker HTTP server (runs in isolated Python 3.11 venv)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import bottle
from bottle import request, response

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

os.chdir(PLUGIN_ROOT)
os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

from config import get_default_model_ckpt, load_config  # noqa: E402
from src.pipeline import (  # noqa: E402
    is_model_loaded,
    load_model,
    run_rig,
    start_bpy_server,
    wait_for_bpy_server,
)

_config = load_config()
WORKER_HOST = _config["worker"]["host"]
WORKER_PORT = _config["worker"]["port"]


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
        import requests

        from src.server.spec import BPY_SERVER

        bpy_ok = False
        try:
            bpy_ok = requests.get(f"{BPY_SERVER}/ping", timeout=1).text == "pong"
        except Exception:
            bpy_ok = False
        return _json_response(
            {
                "status": "ok" if bpy_ok else "degraded",
                "bpy_ready": bpy_ok,
                "model_loaded": is_model_loaded(),
            }
        )

    @app.route("/load_model", method="POST")  # type: ignore[misc]
    def load_model_endpoint():
        try:
            data = _read_json_body()
            model_ckpt = data.get("model_ckpt")
            if not model_ckpt:
                model_ckpt = str(get_default_model_ckpt())
            hf_path = data.get("hf_path")
            message, ckpt = load_model(model_ckpt, hf_path=hf_path)
            return _json_response({"status": "ok", "message": message, "model_ckpt": ckpt})
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb)
            return _json_response({"status": "error", "error": str(exc), "traceback": tb}, status=500)

    @app.route("/infer", method="POST")  # type: ignore[misc]
    def infer_endpoint():
        try:
            data = _read_json_body()
            mesh_path = Path(data["mesh_path"]).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Mesh not found: {mesh_path}")

            output_path = data.get("output_path")
            if output_path:
                out_path = Path(output_path).resolve()
            else:
                out_dir = PLUGIN_ROOT / "output" / "comfyui"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{mesh_path.stem}_rigged.glb"

            model_ckpt = data.get("model_ckpt") or str(get_default_model_ckpt())
            hf_path = data.get("hf_path")
            if hf_path in ("None", ""):
                hf_path = None

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
            return _json_response({"status": "error", "error": str(exc), "traceback": tb}, status=500)

    return app


def main():
    print(f"[TokenRig Worker] plugin root: {PLUGIN_ROOT}")
    start_bpy_server(python=sys.executable, cwd=PLUGIN_ROOT)
    wait_for_bpy_server(timeout=60)

    app = create_app()

    def run_server():
        bottle.run(app, host=WORKER_HOST, port=WORKER_PORT, server="tornado", quiet=False)

    threading.Thread(target=run_server, daemon=False).start()
    print(f"[TokenRig Worker] listening on http://{WORKER_HOST}:{WORKER_PORT}")

    threading.Event().wait()


if __name__ == "__main__":
    main()
