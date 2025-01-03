import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


def swiglu(
    x: torch.Tensor,
    *,
    w3: torch.Tensor,
    w1w2: torch.Tensor | None = None,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    b3: torch.Tensor | None = None,
    b1b2: torch.Tensor | None = None
) -> torch.Tensor:
    assert (
        w1w2 is not None and w1 is None and w2 is None
        or w1w2 is None and w1 is not None and w2 is not None
    )

    if w1w2 is not None:
        x1x2 = F.linear(x, w1w2, b1b2)
        x1, x2 = torch.chunk(x1x2, chunks=2, dim=-1)
    else:
        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
    
    if x.device.type == 'cuda':
        return F.linear(LigerSiLUMulFunction.apply(x1, x2), w3)
    
    return F.linear(F.silu(x1) * x2, w3, b3)


def silu_mul(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if x1.device.type == 'cuda':
        return LigerSiLUMulFunction.apply(x1, x2)
    return F.silu(x1) * x2
