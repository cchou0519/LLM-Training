import logging
import os

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.model_checkpoint import \
    ModelCheckpoint as LightningModelCheckpoint

from llm_training.lightning.loggers.wandb import WandbLogger

logger = logging.getLogger(__name__)


class ModelCheckpoint(LightningModelCheckpoint):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.dirpath is None and isinstance(trainer.logger, WandbLogger):
            self.dirpath = os.path.join(trainer.log_dir, 'checkpoints')

        super().setup(trainer, pl_module, stage)
