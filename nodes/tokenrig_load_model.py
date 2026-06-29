from config import get_default_model_ckpt
from client.worker_client import ensure_worker_running, load_model


class TokenRigLoadModel:
    """Preload TokenRig checkpoint into the worker."""

    @classmethod
    def INPUT_TYPES(cls):
        default_ckpt = str(get_default_model_ckpt())
        return {
            "required": {
                "model_ckpt": ("STRING", {"default": default_ckpt}),
                "hf_path": ("STRING", {"default": "None"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "model_ckpt")
    FUNCTION = "run"
    CATEGORY = "3d/tokenrig"

    def run(self, model_ckpt: str, hf_path: str):
        ensure_worker_running()
        hf = None if hf_path in ("None", "") else hf_path
        message, ckpt = load_model(model_ckpt, hf_path=hf)
        return (message, ckpt)
