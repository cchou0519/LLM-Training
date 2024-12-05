from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.progress.tqdm_progress import \
    TQDMProgressBar as _TQDMProgressBar


class TQDMProgressBar(_TQDMProgressBar):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)

        if trainer.fit_loop.restarting:
            self.train_progress_bar.initial = self.trainer.fit_loop.batch_idx + 1
