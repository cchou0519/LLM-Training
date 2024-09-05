from types import UnionType
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class BaseModelConfig(BaseModel):
    pre_trained_weights: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('*')
    @classmethod
    def validate_torch_dtype(cls, value: Any, info: ValidationInfo) -> Any:
        field = cls.model_fields[info.field_name]
        is_torch_dtype = isinstance(field.annotation, torch.dtype)
        is_torch_dtype |= isinstance(field.annotation, UnionType) and torch.dtype in field.annotation.__args__
        if is_torch_dtype and isinstance(value, str) and value != 'auto':
            value = getattr(torch, value)
        return value
