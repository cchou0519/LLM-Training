from typing import Literal

import torch
from torch import nn


CrossEntropyImplementation = Literal['torch', 'flash_attn']

def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
    label_smoothing: float = 0.0,
    implementation: CrossEntropyImplementation = 'torch'
) -> torch.Tensor:
    if implementation == 'torch':
        return nn.functional.cross_entropy(
            input=logits,
            target=labels,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )

    elif implementation == 'flash_attn':
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

        loss, _ = cross_entropy_loss(
            logits=logits,
            labels=labels,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            inplace_backward=True
        )
        return reduce_loss(
            loss=loss,
            labels=labels,
            ignore_index=ignore_index,
            reduction=reduction
        )

@torch.jit.script
def shift_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    labels = labels.roll(shifts=-1, dims=1)
    index = torch.tensor(-1, device=labels.device)
    labels = labels.index_fill_(1, index, ignore_index)
    return labels


@torch.jit.script
def reduce_loss(
    loss: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    reduction: str
) -> torch.Tensor:
    if reduction == 'mean':
        return loss.sum() / labels.ne(ignore_index).sum()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise NotImplementedError()
