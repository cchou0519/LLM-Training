from dataclasses import field, fields
from types import UnionType

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from llm_training.lr_schedulers import ConstantWarmupLR
from llm_training.utils.config import ConfigBase


class OptimizerConfig(ConfigBase):
    optimizer_class: type[Optimizer]
    optimizer_kwargs: dict
    lr_scheduler_class: type[LRScheduler] = ConstantWarmupLR
    lr_scheduler_kwargs: dict = field(default_factory=dict)


class LitBaseConfig(ConfigBase):
    initialize_weights: bool = False
    pre_trained_weights: str | None = None
    enable_gradient_checkpointing: bool = False
    optim: OptimizerConfig | None = None

    @staticmethod
    def parse_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return dtype
    
    def _parse_torch_dtype(self) -> None:
        for f in fields(self):
            if (
                isinstance(f.type, torch.dtype)
                or isinstance(f.type, UnionType) and torch.dtype in f.type.__args__
            ):
                value = getattr(self, f.name)
                if isinstance(value, str):
                    value = getattr(torch, value)
                setattr(self, f.name, value)

    def __post_init__(self) -> None:
        self._parse_torch_dtype()
