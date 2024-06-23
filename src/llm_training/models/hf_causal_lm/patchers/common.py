import torch
from flash_attn.ops.triton.layer_norm import layer_norm_fn
from torch import nn


def fused_layer_norm_forward(
    self: nn.LayerNorm,
    x: torch.Tensor
) -> torch.Tensor:
    return layer_norm_fn(
        x,
        weight=self.weight,
        bias=self.bias,
        eps=self.eps,   
    )


def clamp_fp16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.half:
        max_dtype = torch.finfo(torch.half).max
        clamp_value = torch.where(torch.isinf(x).any(), max_dtype - 1000, max_dtype)
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x
