"""Bootstrap TokenRig worker venv and optional model download."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from config import get_default_model_ckpt, get_plugin_root, get_venv_dir, get_worker_python, load_config


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print(f"[TokenRig Bootstrap] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def find_python311() -> Optional[Path]:
    env_python = os.environ.get("TOKENRIG_PYTHON")
    if env_python and Path(env_python).is_file():
        return Path(env_python)

    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "python", "find", "3.11"],
                capture_output=True,
                text=True,
                check=True,
            )
            candidate = result.stdout.strip()
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        except subprocess.CalledProcessError:
            pass

    if os.name == "nt" and shutil.which("py"):
        try:
            result = subprocess.run(
                ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                check=True,
            )
            candidate = result.stdout.strip()
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        except subprocess.CalledProcessError:
            pass

    for name in ("python3.11", "python3", "python"):
        candidate = shutil.which(name)
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", "import sys; assert sys.version_info[:2] >= (3, 11)"],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return Path(candidate)
        except OSError:
            continue
    return None


def venv_exists() -> bool:
    return get_worker_python() is not None


def create_venv() -> Path:
    plugin_root = get_plugin_root()
    venv_dir = get_venv_dir()
    if get_worker_python() is not None:
        return get_worker_python()  # type: ignore[return-value]

    python311 = find_python311()
    if python311 is None:
        raise RuntimeError(
            "Python 3.11+ is required for the TokenRig worker. "
            "Install Python 3.11 or set TOKENRIG_PYTHON to its executable."
        )

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if shutil.which("uv"):
            _run(["uv", "venv", str(venv_dir), "--python", str(python311)], cwd=plugin_root)
        else:
            _run([str(python311), "-m", "venv", str(venv_dir)], cwd=plugin_root)
    except subprocess.CalledProcessError as exc:
        hint = ""
        if os.name != "nt":
            hint = (
                " On Debian/Ubuntu, install: sudo apt install python3.11 python3.11-venv"
            )
        raise RuntimeError(f"Failed to create venv at {venv_dir}.{hint}") from exc

    worker_python = get_worker_python()
    if worker_python is None:
        raise RuntimeError(f"Failed to create worker venv at {venv_dir}")
    return worker_python


def install_worker_dependencies() -> None:
    plugin_root = get_plugin_root()
    worker_python = create_venv()
    config = load_config()
    requirements = plugin_root / "worker" / "requirements.txt"
    torch_index = config["torch"]["index_url"]
    torch_version = config["torch"]["version"]

    _run(
        [
            str(worker_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        cwd=plugin_root,
    )
    _run(
        [
            str(worker_python),
            "-m",
            "pip",
            "install",
            f"torch=={torch_version}",
            f"torchvision==0.22.0",
            f"torchaudio=={torch_version}",
            "--index-url",
            torch_index,
        ],
        cwd=plugin_root,
    )
    _run([str(worker_python), "-m", "pip", "install", "-r", str(requirements)], cwd=plugin_root)

    try:
        _run(
            [str(worker_python), "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            cwd=plugin_root,
        )
    except subprocess.CalledProcessError:
        print("[TokenRig Bootstrap] flash-attn install failed; worker will fall back to sdpa.")


def download_default_model() -> None:
    plugin_root = get_plugin_root()
    worker_python = get_worker_python()
    if worker_python is None:
        worker_python = create_venv()

    ckpt_path = get_default_model_ckpt()
    if ckpt_path.is_file():
        print(f"[TokenRig Bootstrap] Model already present: {ckpt_path}")
        return

    print("[TokenRig Bootstrap] Downloading default model checkpoint...")
    _run([str(worker_python), str(plugin_root / "download.py"), "--model"], cwd=plugin_root)


def ensure_worker_installed(force: bool = False) -> str:
    config = load_config()
    messages: List[str] = []
    venv_dir = get_venv_dir()

    if force and venv_dir.exists():
        messages.append(f"Removing existing worker venv: {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)

    if force or not venv_exists():
        messages.append("Creating worker virtual environment...")
        create_venv()
        messages.append("Installing worker dependencies (this may take several minutes)...")
        install_worker_dependencies()
    else:
        messages.append(f"Worker venv ready: {venv_dir}")

    if config["model"].get("auto_download", True):
        try:
            download_default_model()
            messages.append("Model checkpoint ready.")
        except subprocess.CalledProcessError as exc:
            messages.append(f"Model download failed: {exc}. Run download.py --model manually.")

    return "\n".join(messages)


def main(argv: Optional[Iterable[str]] = None) -> int:
    force = "--force" in list(argv or sys.argv[1:])
    print(ensure_worker_installed(force=force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
