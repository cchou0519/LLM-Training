import time
from typing import Any, Mapping

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor


class TrainingTimeEstimator(Callback):
    def __init__(
        self,
        num_test_steps: int,
        num_warmup_steps: int = 2,
    ) -> None:
        super().__init__()

        assert num_warmup_steps >= 0
        assert num_warmup_steps < num_test_steps
        
        self.num_test_steps = num_test_steps
        self.num_warmup_steps = num_warmup_steps

        self.start_time = 0
        self.end_time = 0

    def print_info(self):
        seconds = self.end_time - self.start_time
        steps_per_second = (self.num_test_steps - self.num_warmup_steps) / seconds
        estimated_seconds = self.num_total_steps / steps_per_second
        print(f'Running seconds: {seconds}')
        print(f'Steps per second: {steps_per_second:.2f}')
        print(f'Estimated total seconds: {estimated_seconds:.2f}')
        print(f'Estimated total minutes: {estimated_seconds / 60:.2f}')
        print(f'Estimated total hours: {estimated_seconds / 60 ** 2:.2f}')

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if trainer.global_step == self.num_warmup_steps:
            self.start_time = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if trainer.global_step == self.num_test_steps:
            self.end_time = time.time()
            trainer.should_stop = True

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.num_total_steps = trainer.estimated_stepping_batches

        if trainer.is_global_zero:
            self.print_info()
