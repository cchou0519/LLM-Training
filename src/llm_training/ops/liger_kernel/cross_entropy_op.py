from typing import Literal

import torch
import torch.nn.functional as F
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import \
    LigerFusedLinearCrossEntropyFunction


def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: Literal['mean'] = 'mean'
) -> torch.Tensor:
    assert reduction == 'mean'

    if logits.dim() == 3 and labels.dim() == 2:
        logits = logits.flatten(end_dim=1)
        labels = labels.flatten(end_dim=1)
    
    if logits.device.type != 'cuda':
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index
        )
    
    return LigerCrossEntropyFunction.apply(
        logits,
        labels,
        ignore_index
    )


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: Literal['mean'] = 'mean'
) -> torch.Tensor:
    assert reduction == 'mean'

    if hidden_states.dim() == 3 and labels.dim() == 2:
        hidden_states = hidden_states.flatten(end_dim=1)
        labels = labels.flatten(end_dim=1)

    return LigerFusedLinearCrossEntropyFunction.apply(
        hidden_states,
        weight,
        labels,
        ignore_index
    )
