"""Plugin configuration (no heavy dependencies)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

PLUGIN_ROOT = Path(__file__).resolve().parent
VENV_DIR_NAME = ".tokenrig-venv"
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 59877
BPY_PORT = 59876

DEFAULT_MODEL_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_HF_PATH = None

DEFAULTS: Dict[str, Any] = {
    "worker": {
        "host": WORKER_HOST,
        "port": WORKER_PORT,
        "venv": VENV_DIR_NAME,
        "auto_start": True,
        "auto_install": True,
        "startup_timeout": 300,
    },
    "model": {
        "auto_download": True,
        "default_ckpt": DEFAULT_MODEL_CKPT,
        "default_hf_path": DEFAULT_HF_PATH,
    },
    "torch": {
        "index_url": "https://download.pytorch.org/whl/cu128",
        "version": "2.7.0",
    },
}


def get_plugin_root() -> Path:
    return PLUGIN_ROOT


def get_config_path() -> Path:
    return PLUGIN_ROOT / "config.json"


def load_config() -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULTS))
    config_path = get_config_path()
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            user_config = json.load(f)
        _deep_merge(config, user_config)
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_venv_dir(config: Dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    venv_name = config["worker"]["venv"]
    venv_path = Path(venv_name)
    if not venv_path.is_absolute():
        venv_path = PLUGIN_ROOT / venv_path
    return venv_path


def get_worker_python(config: Dict[str, Any] | None = None) -> Path | None:
    env_python = os.environ.get("TOKENRIG_PYTHON")
    if env_python:
        path = Path(env_python)
        if path.is_file():
            return path

    venv_dir = get_venv_dir(config)
    if os.name == "nt":
        candidates = [venv_dir / "Scripts" / "python.exe"]
    else:
        candidates = [venv_dir / "bin" / "python", venv_dir / "bin" / "python3"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def get_worker_url(config: Dict[str, Any] | None = None) -> str:
    config = config or load_config()
    host = config["worker"]["host"]
    port = config["worker"]["port"]
    return f"http://{host}:{port}"


def get_default_model_ckpt(config: Dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    ckpt = config["model"]["default_ckpt"]
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PLUGIN_ROOT / ckpt_path
    return ckpt_path
