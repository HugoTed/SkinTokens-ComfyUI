import os
import tempfile
from pathlib import Path

from config import get_default_model_ckpt
from client.worker_client import ensure_worker_running, infer

_LOAD_STATUS_MESSAGES = frozenset({"Model loaded.", "Model already loaded."})

try:
    from comfy_api.latest import Types as ComfyTypes
except ImportError:
    ComfyTypes = None

SUPPORTED_MESH_EXT = {".obj", ".fbx", ".glb"}
_MESH_INPUT_TYPES = "STRING,FILE_3D,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_FBX,FILE_3D_STL"


def _temp_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_temp_directory()
    except ImportError:
        return tempfile.gettempdir()


def _validate_mesh_ext(path: Path) -> None:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_MESH_EXT:
        supported = ", ".join(sorted(SUPPORTED_MESH_EXT))
        raise ValueError(f"Unsupported mesh format: {ext or '(none)'}. TokenRig supports: {supported}")


def _resolve_file3d(file3d) -> str:
    if file3d.is_disk_backed:
        path = Path(file3d.get_source()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input mesh not found: {path}")
        _validate_mesh_ext(path)
        return str(path)

    ext = file3d.format or "glb"
    if not ext.startswith("."):
        ext = f".{ext}"
    _validate_mesh_ext(Path(f"model{ext}"))

    fd, tmp_path = tempfile.mkstemp(
        suffix=ext,
        prefix="tokenrig_input_",
        dir=_temp_dir(),
    )
    os.close(fd)
    file3d.save_to(tmp_path)
    return tmp_path


def _resolve_mesh_path(path_str: str) -> str:
    path_str = path_str.strip()
    if not path_str:
        raise ValueError(
            "mesh_path is empty. Connect Load 3D mesh_path/model_3d or set a file path."
        )

    try:
        import folder_paths

        if folder_paths.exists_annotated_filepath(path_str):
            resolved = folder_paths.get_annotated_filepath(path_str)
            path = Path(resolved).resolve()
            if path.is_file():
                _validate_mesh_ext(path)
                return str(path)
    except ImportError:
        pass

    path = Path(path_str)
    if path.is_file():
        _validate_mesh_ext(path.resolve())
        return str(path.resolve())
    raise FileNotFoundError(f"Input mesh not found: {path_str}")


def _resolve_mesh_input(mesh_input) -> str:
    if ComfyTypes is not None and isinstance(mesh_input, ComfyTypes.File3D):
        return _resolve_file3d(mesh_input)
    return _resolve_mesh_path(str(mesh_input))


def _file3d_from_path(path: str, export_format: str):
    if ComfyTypes is None:
        raise RuntimeError(
            "ComfyUI File3D types are not available. "
            "Upgrade ComfyUI to a version with built-in 3D support (Save 3D Model / Preview 3D)."
        )
    return ComfyTypes.File3D(path, file_format=export_format)


class TokenRigGenerate:
    """Generate rigged mesh (GLB or FBX) from an input mesh via the TokenRig worker."""

    @classmethod
    def INPUT_TYPES(cls):
        default_ckpt = str(get_default_model_ckpt())
        mesh_type = _MESH_INPUT_TYPES if ComfyTypes is not None else "STRING"
        return {
            "required": {
                "mesh_path": (mesh_type, {"default": "", "multiline": False}),
                "model_ckpt": ("STRING", {"default": default_ckpt}),
                "hf_path": ("STRING", {"default": "None"}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "export_format": (["glb", "fbx"], {"default": "glb"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
                "use_skeleton": ("BOOLEAN", {"default": False}),
                "use_transfer": ("BOOLEAN", {"default": False}),
                "use_postprocess": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "setup_status": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FILE_3D_GLB")
    RETURN_NAMES = ("output_path", "status", "model_3d")
    FUNCTION = "run"
    CATEGORY = "3d/tokenrig"
    OUTPUT_NODE = True

    def run(
        self,
        mesh_path: str,
        model_ckpt: str,
        hf_path: str,
        output_path: str,
        export_format: str,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        num_beams: int,
        use_skeleton: bool,
        use_transfer: bool,
        use_postprocess: bool,
        setup_status: str = "",
    ):
        mesh = _resolve_mesh_input(mesh_path)

        ensure_worker_running()

        out = output_path.strip() or None
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        hf = None if hf_path in ("None", "") else hf_path
        ckpt = model_ckpt.strip()
        if ckpt in _LOAD_STATUS_MESSAGES or not ckpt:
            ckpt = str(get_default_model_ckpt())
        result_path = infer(
            mesh_path=mesh,
            output_path=out,
            model_ckpt=ckpt,
            hf_path=hf,
            export_format=export_format,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            use_skeleton=use_skeleton,
            use_transfer=use_transfer,
            use_postprocess=use_postprocess,
        )
        status = f"Rigged {export_format.upper()} saved to: {result_path}"
        if setup_status:
            status = f"{setup_status}\n{status}"
        model_3d = _file3d_from_path(str(Path(result_path).resolve()), export_format)
        return (result_path, status, model_3d)
