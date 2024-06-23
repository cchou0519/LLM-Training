from typing import Any

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class WarmupLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        num_warmup_epochs: int,
        last_epoch: int = -1
    ) -> None:
        self.lr_scheduler = lr_scheduler
        self.num_warmup_epochs = num_warmup_epochs
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch >= self.num_warmup_epochs:
            return self.lr_scheduler.get_lr()
        return [(self.last_epoch + 1) / self.num_warmup_epochs * lr for lr in self.base_lrs]
    
    def step(self, epoch: int | None = None) -> None:
        if self.last_epoch == self.num_warmup_epochs:
            self.lr_scheduler.base_lrs = self.base_lrs

        if self.last_epoch >= self.num_warmup_epochs:
            epoch = None if epoch is None else epoch - self.num_warmup_epochs
            self.lr_scheduler.step(epoch)
            self._last_lr = self.lr_scheduler.get_last_lr()
        return super().step(epoch)

    def state_dict(self) -> dict[str, Any]:
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ['optimizer', 'lr_scheduler']}
        state_dict['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        lr_scheduler_state_dict = state_dict.pop('lr_scheduler_state_dict')
        self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        super().load_state_dict(state_dict)
