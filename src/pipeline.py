"""TokenRig inference pipeline shared by demo, worker, and CLI."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from torch import Tensor
from tqdm import tqdm

from process_util import popen_detached, terminate_process_tree

from .data.dataset import DatasetConfig, RigDatasetModule
from .data.transform import Transform
from .data.vertex_group import voxel_skin
from .model.tokenrig import TokenRigResult
from .server.spec import BPY_SERVER, get_model, object_to_bytes, bytes_to_object
from .tokenizer.parse import get_tokenizer

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

SUPPORTED_EXT = {".obj", ".fbx", ".glb"}
SUPPORTED_EXPORT_EXT = {".glb", ".fbx"}


def normalize_export_format(export_format: str) -> str:
    fmt = export_format.lower().lstrip(".")
    if fmt not in ("glb", "fbx"):
        raise ValueError(f"Unsupported export format: {export_format!r}. Use glb or fbx.")
    return fmt


def export_suffix(export_format: str) -> str:
    return f".{normalize_export_format(export_format)}"


def resolve_output_path(
    mesh_path: Path,
    output_path: Optional[Path],
    export_format: str = "glb",
    default_dir: Optional[Path] = None,
) -> Path:
    suffix = export_suffix(export_format)
    if output_path is None:
        out_dir = default_dir or mesh_path.parent
        return out_dir / f"{mesh_path.stem}_rigged{suffix}"
    output_path = Path(output_path)
    if output_path.suffix.lower() in SUPPORTED_EXPORT_EXT:
        return output_path
    if output_path.suffix:
        return output_path
    return output_path.with_suffix(suffix)

model = None
tokenizer = None
transform = None
CURRENT_MODEL_CKPT: Optional[str] = None
CURRENT_HF_PATH: Optional[str] = None

_LOAD_STATUS_MESSAGES = frozenset({"Model loaded.", "Model already loaded."})

_bpy_proc = None


def resolve_model_ckpt(model_ckpt: Optional[str]) -> str:
    """Map ComfyUI status strings / empty values to a real checkpoint path."""
    from config import get_default_model_ckpt

    ckpt = (model_ckpt or "").strip()
    if ckpt in _LOAD_STATUS_MESSAGES or not ckpt:
        if CURRENT_MODEL_CKPT:
            return CURRENT_MODEL_CKPT
        return str(get_default_model_ckpt())
    return ckpt


def get_plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _bpy_log_path(plugin_root: Path) -> Path:
    return plugin_root / "tokenrig-bpy.log"


def start_bpy_server(python: Optional[str] = None, cwd: Optional[Path] = None):
    global _bpy_proc
    if _bpy_proc is not None and _bpy_proc.poll() is None:
        return _bpy_proc

    plugin_root = cwd or get_plugin_root()
    executable = python or sys.executable
    log_path = _bpy_log_path(plugin_root)
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n--- bpy_server start {datetime.now().isoformat()} ---\n")
    log_file.flush()
    proc = popen_detached(
        [executable, str(plugin_root / "bpy_server.py")],
        cwd=str(plugin_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"[TokenRig] bpy_server started (pid={proc.pid}), log: {log_path}")
    _bpy_proc = proc

    def cleanup():
        print(f"[TokenRig] Terminating bpy_server (pid={proc.pid})")
        terminate_process_tree(proc)

    atexit.register(cleanup)
    return proc


def wait_for_bpy_server(timeout: float = 30) -> None:
    t0 = time.time()
    log_path = _bpy_log_path(get_plugin_root())
    while True:
        if _bpy_proc is not None and _bpy_proc.poll() is not None:
            tail = ""
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-50:])
            raise RuntimeError(
                "bpy_server process exited during startup.\n"
                f"Last lines from {log_path}:\n{tail}\n"
                "Tip: run `.tokenrig-venv/bin/python bpy_server.py` in a terminal."
            )
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            print("[TokenRig] bpy_server is ready")
            return
        except Exception:
            if time.time() - t0 > timeout:
                tail = ""
                if log_path.is_file():
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = "\n".join(lines[-50:])
                raise RuntimeError(
                    f"bpy_server failed to start within {timeout:.0f}s.\n"
                    f"Last lines from {log_path}:\n{tail}\n"
                    "On Linux install: sudo apt install libgl1 libglib2.0-0 libxrender1 libxext6 libxkbcommon0"
                )
            time.sleep(0.5)


def load_model(model_ckpt: str, hf_path: Optional[str] = None) -> Tuple[str, str]:
    global model, tokenizer, transform, CURRENT_MODEL_CKPT, CURRENT_HF_PATH
    from config import normalize_hf_path

    hf_path = normalize_hf_path(hf_path)
    model_ckpt = resolve_model_ckpt(model_ckpt)
    if model is not None and model_ckpt == CURRENT_MODEL_CKPT and hf_path == CURRENT_HF_PATH:
        return ("Model already loaded.", model_ckpt)

    if not model_ckpt:
        raise RuntimeError("model_ckpt is empty. Please select a checkpoint.")

    print(f"[TokenRig] Loading model: {model_ckpt}, hf_path={hf_path}")
    model = get_model(model_ckpt, hf_path=hf_path)
    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    CURRENT_MODEL_CKPT = model_ckpt
    CURRENT_HF_PATH = hf_path
    return ("Model loaded.", model_ckpt)


def is_model_loaded() -> bool:
    return model is not None


def post_bpy_payload(endpoint: str, payload):
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        request_payload = {"payload_path": payload_path}
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes(request_payload),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


def run_rig(
    filepaths: List[Path],
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    num_beams: int,
    use_skeleton: bool,
    use_transfer: bool,
    use_postprocess: bool,
    output_paths: List[Path],
    model_ckpt: str,
    hf_path: Optional[str],
) -> List[Path]:
    assert len(filepaths) == len(output_paths)

    load_model(resolve_model_ckpt(model_ckpt), hf_path)

    datapath = {
        "data_name": None,
        "loader": "bpy_server",
        "filepaths": {"articulation": [str(p) for p in filepaths]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=transform,
        tokenizer=tokenizer,
        process_fn=model._process_fn,
    )

    dataloader = module.predict_dataloader()["articulation"]
    results_out = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {
            k: v.to("cuda") if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        if not use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = dict(
            max_length=2048,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=1,
            num_beams=int(num_beams),
            do_sample=True,
        )

        if "skeleton_tokens" in batch and "skeleton_mask" in batch:
            mask = batch["skeleton_mask"][0] == 1
            skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
        else:
            skeleton_tokens = None

        preds: List[TokenRigResult] = model.predict_step(
            batch,
            skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
            make_asset=True,
        )["results"]

        asset = preds[0].asset
        assert asset is not None

        if use_postprocess:
            voxel = asset.voxel(resolution=196)
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()

        out_path = output_paths[i]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_transfer:
            payload = dict(
                source_asset=asset,
                target_path=asset.path,
                export_path=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("transfer", payload)
        else:
            payload = dict(
                asset=asset,
                filepath=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("export", payload)

        if res != "ok":
            print(f"[TokenRig Error] {res}")
        else:
            print(f"[TokenRig OK] Exported: {out_path}")

        results_out.append(out_path)

    return results_out


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


def map_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
    export_format: str = "glb",
) -> Path:
    rel = in_path.relative_to(input_root)
    return (output_root / rel).with_suffix(export_suffix(export_format))
