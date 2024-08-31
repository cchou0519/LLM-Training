import torch
from liger_kernel.ops.rope import LigerRopeFunction


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return LigerRopeFunction.apply(
        q,
        k,
        cos,
        sin
    )
