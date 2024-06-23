
from typing import Literal

import torch


RMSNormImplementation = Literal['torch', 'flash_attn']

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    implementation: RMSNormImplementation = 'torch'
) -> torch.Tensor:
    if implementation == 'torch':
        return rms_norm_torch(x, weight=weight, eps=eps)
 
    elif implementation == 'flash_attn':
        from flash_attn.ops.triton.layer_norm import rms_norm_fn

        return rms_norm_fn(
            x,
            weight=weight,
            bias=None,
            eps=eps
        )
    
    else:
        raise NotImplementedError()


@torch.jit.script
def rms_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float
) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)
