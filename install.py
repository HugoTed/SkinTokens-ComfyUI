"""ComfyUI Manager install hook for TokenRig."""

from bootstrap import ensure_worker_installed


if __name__ == "__main__":
    print(ensure_worker_installed())
