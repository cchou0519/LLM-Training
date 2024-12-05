import torch


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float
) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x.to(dtype)
    return x
