import torch


def shift_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    labels = labels.roll(shifts=-1, dims=1)
    index = torch.tensor(-1, device=labels.device)
    labels = labels.index_fill_(1, index, ignore_index)
    return labels
