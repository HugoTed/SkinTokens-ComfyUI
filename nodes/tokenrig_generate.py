from pathlib import Path

from config import get_default_model_ckpt
from client.worker_client import ensure_worker_running, infer


class TokenRigGenerate:
    """Generate rigged mesh (GLB or FBX) from an input mesh via the TokenRig worker."""

    @classmethod
    def INPUT_TYPES(cls):
        default_ckpt = str(get_default_model_ckpt())
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False}),
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "status")
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
        mesh = Path(mesh_path.strip())
        if not mesh.is_file():
            raise FileNotFoundError(f"Input mesh not found: {mesh}")

        ensure_worker_running()

        out = output_path.strip() or None
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        hf = None if hf_path in ("None", "") else hf_path
        result_path = infer(
            mesh_path=str(mesh),
            output_path=out,
            model_ckpt=model_ckpt,
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
        return (result_path, status)
