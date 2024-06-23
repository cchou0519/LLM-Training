from dataclasses import KW_ONLY, field
from enum import auto

from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based import HFBasedDataModuleConfig
from llm_training.utils.str_enum import StrEnum


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    CONCAT_AND_TRUNCATE = auto()


class PreTrainingDataModuleConfig(HFBasedDataModuleConfig):
    _: KW_ONLY
    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None
    stride: int | None = None
    concat_method: ConcatMethod | str = ConcatMethod.CONCAT_AND_TRUNCATE
    pad_to_multiple_of: int | None = None
    sample_rate: dict[str, float] = field(default_factory=dict)
    shuffle_before_tokenization: bool = False
    concat_and_truncate_batch_size: int = 100000

    def __post_init__(self):
        super().__post_init__()

        self.concat_method = ConcatMethod(self.concat_method.lower())

        if self.concat_method == ConcatMethod.CONCAT_AND_TRUNCATE:
            assert self.max_length is not None, f"You must set `max_length` to use `CONCAT_AND_TRUNCATE`"

        if self.stride is not None:
            assert self.max_length is not None, "You must also set `max_length` to use `stride`"
            assert self.stride <= self.max_length, "`stride` must be <= `max_length`"
        else:
            self.stride = self.max_length
