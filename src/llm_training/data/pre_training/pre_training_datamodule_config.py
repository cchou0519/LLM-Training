from enum import auto

from pydantic import Field, ValidationInfo, field_validator
from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based import HFBasedDataModuleConfig
from llm_training.utils.str_enum import StrEnum


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    CONCAT_AND_TRUNCATE = auto()


class PreTrainingDataModuleConfig(HFBasedDataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None
    stride: int | None = None
    concat_method: ConcatMethod | str = ConcatMethod.CONCAT_AND_TRUNCATE
    pad_to_multiple_of: int | None = None
    sample_rate: dict[str, float] = Field(default_factory=dict)
    shuffle_before_tokenization: bool = False
    concat_and_truncate_batch_size: int = 100000

    @field_validator('concat_method')
    @classmethod
    def validate_concat_method(cls, value: ConcatMethod | str, info: ValidationInfo) -> ConcatMethod:
        value = ConcatMethod(value.lower())
        assert value != ConcatMethod.CONCAT_AND_TRUNCATE or info.data['max_length'] is not None, \
            "You must set `max_length` to use `CONCAT_AND_TRUNCATE`"
        return value
    
    @field_validator('stride')
    @classmethod
    def validate_stride(cls, value: int | None, info: ValidationInfo) -> int | None:
        max_length = info.data['max_length']

        if value is None:
            value = max_length
        else:
            assert max_length is not None, "You must also set `max_length` to use `stride`"
            assert value <= max_length, "`stride` must be <= `max_length`"

        return value
