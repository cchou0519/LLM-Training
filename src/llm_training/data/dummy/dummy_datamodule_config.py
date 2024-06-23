import random
from dataclasses import KW_ONLY

from llm_training.data.base_datamodule_config import BaseDataModuleConfig


class DummyDataModuleConfig(BaseDataModuleConfig):
    _: KW_ONLY
    vocab_size: int
    max_length: int
    num_samples: int | None = None
    num_tokens: int | None = None
    base_seed: int | None = None

    def __post_init__(self):
        super().__post_init__()

        assert self.num_samples is not None or self.num_tokens is not None

        if self.base_seed is None:
            self.base_seed = random.randrange(0, 999999)
