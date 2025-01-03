import torch
from torchmetrics.functional.text.perplexity import (_perplexity_compute,
                                                     _perplexity_update)

from .metric import Metric


class Perplexity(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    total_log_probs: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        ignore_index: int | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(f"Argument `ignore_index` expected to either be `None` or an `int` but got {ignore_index}")

        self.ignore_index = ignore_index
        self.add_state('total_log_probs', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, preds_or_loss: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if preds_or_loss.dim() == 0:
            self.total_log_probs += preds_or_loss
            self.count += 1
        else:
            total_log_probs, count = _perplexity_update(preds_or_loss, target, self.ignore_index)
            self.total_log_probs += total_log_probs
            self.count += count

    def compute(self) -> torch.Tensor:
        return _perplexity_compute(self.total_log_probs, self.count)
