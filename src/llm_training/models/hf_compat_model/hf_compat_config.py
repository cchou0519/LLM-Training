from typing import Literal

import torch
from pydantic import Field

from llm_training.models.base_model.base_model_config import BaseModelConfig


class HFCompatModelConfig(BaseModelConfig):
    hf_path: str | None = None
    hf_tokenizer_path: str | None = None

    torch_dtype: str | torch.dtype = 'auto'
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    revision: str = 'main'
    attn_implementation: Literal['eager', 'sdpa', 'flash_attention_2'] | None = None
    hf_extra_kwargs: dict = Field(default_factory=dict)

    load_hf_weights: bool = True

    @property
    def _attn_implementation(self) -> str:
        if self.attn_implementation is None:
            return 'flash_attention_2' if torch.cuda.get_device_capability()[0] >= 8 else 'sdpa'
        return self.attn_implementation
