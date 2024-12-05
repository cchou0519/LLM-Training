import torch
from liger_kernel.ops.rope import LigerRopeFunction

from llm_training.ops.rope_op import apply_rope as apply_rope_torch


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.device.type != 'cuda':
        return apply_rope_torch(q, k, cos, sin)
    
    return LigerRopeFunction.apply(
        q,
        k,
        cos,
        sin
    )
