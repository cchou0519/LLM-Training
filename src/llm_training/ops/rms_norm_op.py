import torch
from flash_attn.ops.triton.layer_norm import rms_norm_fn


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float
) -> torch.Tensor:
    return rms_norm_fn(
        x,
        weight=weight,
        bias=None,
        eps=eps
    )
