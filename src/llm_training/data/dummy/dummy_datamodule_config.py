import random

from pydantic import Field, ValidationInfo, field_validator

from llm_training.data.base_datamodule_config import BaseDataModuleConfig


class DummyDataModuleConfig(BaseDataModuleConfig):
    vocab_size: int
    max_length: int
    num_samples: int | None = None
    num_tokens: int | None = None
    base_seed: int | None = Field(None, validate_default=True)
    
    @field_validator('num_tokens')
    @classmethod
    def validate_num_tokens(cls, value: int | None, info: ValidationInfo):
        assert info.data['num_samples'] is None
        return value

    @field_validator('base_seed')
    @classmethod
    def validate_base_seed(cls, value: int | None):
        if value is None:
            value = random.randrange(0, 999999)
        return value
