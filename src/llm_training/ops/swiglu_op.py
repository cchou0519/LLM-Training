from typing import Literal

import torch
import torch.nn.functional as F

SwiGLUImplementation = Literal['eager', 'flash_attn', 'xformers']


def swiglu(
    x: torch.Tensor,
    w1w2: torch.Tensor,
    w3: torch.Tensor,
    implementation: SwiGLUImplementation
) -> torch.Tensor:
    
    if implementation == 'eager':
        x1x2 = F.linear(x, w1w2)
        x1, x2 = torch.chunk(x1x2, chunks=2, dim=-1)
        return F.linear(F.silu(x1) * x2, w3)
    elif implementation == 'flash_attn':
        from flash_attn.ops.activations import swiglu

        x1x2 = F.linear(x, w1w2)
        x1, x2 = torch.chunk(x1x2, chunks=2, dim=-1)
        return F.linear(swiglu(x1, x2), w3)
    elif implementation == 'xformers':
        from xformers.ops.swiglu_op import swiglu, unbind # type: ignore

        w1, w2 = unbind(w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]), dim=0)
        return swiglu(
            x,
            w1=w1,
            b1=None,
            w2=w2,
            b2=None,
            w3=w3,
            b3=None
        )
    else:
        raise NotImplementedError()
