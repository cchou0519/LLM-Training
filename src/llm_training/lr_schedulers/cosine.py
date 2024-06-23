from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from .warmup import WarmupLR


class CosineAnnealingWarmupLR(WarmupLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_total_steps: int,
        min_lr: float,
        last_epoch: int = -1
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=CosineAnnealingLR(
                optimizer=optimizer,
                T_max=num_total_steps - num_warmup_steps,
                eta_min=min_lr,
                last_epoch=last_epoch
            ),
            num_warmup_epochs=num_warmup_steps,
            last_epoch=last_epoch
        )
