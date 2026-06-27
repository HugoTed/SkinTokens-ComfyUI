from bootstrap import ensure_worker_installed
from client.worker_client import ensure_worker_running, ping_worker


class TokenRigSetup:
    """Install worker venv, download model, and start the TokenRig worker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_reinstall": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "3d/tokenrig"

    def run(self, force_reinstall: bool):
        messages = []
        try:
            messages.append(ensure_worker_installed(force=force_reinstall))
            messages.append(ensure_worker_running())
            health = "ready" if ping_worker() else "not ready"
            messages.append(f"Worker health: {health}")
        except Exception as exc:
            messages.append(f"Setup failed: {exc}")
            raise RuntimeError("\n".join(messages)) from exc
        return ("\n".join(messages),)
