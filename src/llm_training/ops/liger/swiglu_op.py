import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


def swiglu(
    x: torch.Tensor,
    *,
    w3: torch.Tensor,
    w1w2: torch.Tensor | None = None,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None
) -> torch.Tensor:
    assert (
        w1w2 is not None and w1 is None and w2 is None
        or w1w2 is None and w1 is not None and w2 is not None
    )

    if w1w2 is not None:
        x1x2 = F.linear(x, w1w2)
        x1, x2 = torch.chunk(x1x2, chunks=2, dim=-1)
    else:
        x1 = F.linear(x, w1)
        x2 = F.linear(x, w2)
    
    return F.linear(LigerSiLUMulFunction.apply(x1, x2), w3)
