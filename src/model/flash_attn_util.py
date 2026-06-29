"""Flash attention helpers with PyTorch SDPA fallback."""

from __future__ import annotations

import torch.nn.functional as F

_USE_FLASH_ATTN = False


try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore

    _USE_FLASH_ATTN = True

    def flash_attn_func(q, k, v, **kwargs):
        res = _flash_attn_func(q, k, v, **kwargs)
        if isinstance(res, tuple):
            return res
        return res, None
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore

        _USE_FLASH_ATTN = True

        def flash_attn_func(q, k, v, **kwargs):
            res = _flash_attn_func(q, k, v, **kwargs)
            if isinstance(res, tuple):
                return res
            return res, None
    except ImportError:
        def flash_attn_func(q, k, v, **kwargs):
            # q, k, v: (B, L, H, D) -> SDPA expects (B, H, L, D)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            if q_t.shape[1] != k_t.shape[1]:
                repeat_factor = q_t.shape[1] // k_t.shape[1]
                k_t = k_t.repeat_interleave(repeat_factor, dim=1)
                v_t = v_t.repeat_interleave(repeat_factor, dim=1)
            out = F.scaled_dot_product_attention(q_t, k_t, v_t)
            return out.transpose(1, 2), None


def get_transformers_attn_implementation() -> str:
    """Return flash_attention_2 when flash-attn is installed, else PyTorch SDPA."""
    return "flash_attention_2" if _USE_FLASH_ATTN else "sdpa"
