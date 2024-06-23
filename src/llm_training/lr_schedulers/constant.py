from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR

from .warmup import WarmupLR


class ConstantWarmupLR(WarmupLR):
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0,
        total_iters: int = 0,
        num_warmup_steps: int = 0,
        last_epoch: int = -1
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=ConstantLR(
                optimizer=optimizer,
                factor=factor,
                total_iters=total_iters,
                last_epoch=last_epoch
            ),
            num_warmup_epochs=num_warmup_steps,
            last_epoch=last_epoch
        )
