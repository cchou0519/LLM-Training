import logging
import os
from typing import Literal
from uuid import uuid4

from lightning import LightningModule, Trainer
import torch
from lightning.pytorch.callbacks.callback import Callback
from triton.runtime.cache import default_cache_dir

logger = logging.getLogger(__name__)


class ExtraConfig(Callback):
    def __init__(
        self,
        float32_matmul_precision: Literal["medium", "high", "highest"] | None = None,
        logging_level: int | str = logging.INFO
    ) -> None:
        super().__init__()

        self.float32_matmul_precision = float32_matmul_precision
        self.logging_level = logging_level

        self._configure_float32_matmul_precision()
        self._configure_logging_level()
        
    def _configure_float32_matmul_precision(self) -> None:
        if self.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self.float32_matmul_precision)

    def _configure_logging_level(self) -> None:
        if isinstance(self.logging_level, str):
            logging_level = getattr(logging, self.logging_level.upper())
        else:
            logging_level = self.logging_level

        logging.getLogger('llm_training').setLevel(logging_level)
        logging.getLogger('lightning').setLevel(logging_level)

    def _configure_triton_cache_dir(self) -> None:
        if not os.getenv('TRITON_CACHE_DIR', '').strip():
            os.environ['TRITON_CACHE_DIR'] = os.path.join(default_cache_dir(), str(uuid4()))

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        self._configure_triton_cache_dir()
