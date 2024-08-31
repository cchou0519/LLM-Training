import torch
from liger_kernel.ops.rms_norm import LigerRMSNormFunction


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float
) -> torch.Tensor:
    return LigerRMSNormFunction.apply(
        x,
        weight,
        eps
    )
