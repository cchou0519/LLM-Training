import torch
from liger_kernel.ops.rms_norm import LigerRMSNormFunction

from llm_training.ops.rms_norm_op import rms_norm as rms_norm_torch


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float
) -> torch.Tensor:
    if x.device.type != 'cuda':
        return rms_norm_torch(x, weight, eps)

    return LigerRMSNormFunction.apply(
        x,
        weight,
        eps
    )
