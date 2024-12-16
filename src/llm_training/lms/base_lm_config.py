from types import UnionType
from typing import Any

import torch
from pydantic import BaseModel as PyDanticBaseModel
from pydantic import ConfigDict, Field, ValidationInfo, field_validator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from llm_training.lr_schedulers import ConstantWarmupLR


class BaseOptimizerConfig(PyDanticBaseModel):
    optimizer_class: type[Optimizer]
    optimizer_kwargs: dict[str, Any]
    lr_scheduler_class: type[LRScheduler] = ConstantWarmupLR
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseLightningModuleConfig(PyDanticBaseModel):
    init_weights: bool = False
    load_weights: bool = True
    optim: BaseOptimizerConfig | None = None
    frozen_modules: list[str] | None = None
    log_grad_norm: bool = True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    @field_validator('*')
    @classmethod
    def validate_torch_dtype(cls, value: Any, info: ValidationInfo) -> Any:
        field = cls.model_fields[info.field_name]
        is_torch_dtype = isinstance(field.annotation, torch.dtype)
        is_torch_dtype |= isinstance(field.annotation, UnionType) and torch.dtype in field.annotation.__args__
        if is_torch_dtype and isinstance(value, str) and value != 'auto':
            value = getattr(torch, value)
        return value
