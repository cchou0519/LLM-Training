from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_total_steps: int,
        min_lr: float | list[float],
        last_epoch: int = -1
    ) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_total_steps = num_total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

        assert isinstance(self.min_lr, float) or len(min_lr) == len(self.base_lrs)

    @property
    def min_lrs(self) -> list[float]:
        if isinstance(self.min_lr, float):
            return [self.min_lr] * len(self.base_lrs)
        return self.min_lr

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [(self.last_epoch + 1) / (self.num_warmup_steps + 1) * lr for lr in self.base_lrs]
        
        lrs = []
        for lr, min_lr in zip(self.base_lrs, self.min_lrs):
            factor = (self.num_total_steps - self.last_epoch) / (self.num_total_steps - self.num_warmup_steps)
            min_lr_factor = min_lr / lr
            factor = (1.0 - min_lr_factor) * (factor - 0.0) + min_lr_factor
            lrs.append(lr * factor)

        return lrs
