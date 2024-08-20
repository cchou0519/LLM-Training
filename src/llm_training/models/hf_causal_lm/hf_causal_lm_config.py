from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import GenerationConfig

from llm_training.models.hf_compat_model import HFCompatModelConfig

if TYPE_CHECKING:
    try:
        from peft import PeftConfig  # type: ignore
    except ImportError: ...
else:
    try:
        from peft import PeftConfig
    except ImportError:
        PeftConfig = dict[str, Any]


try:
    from peft import get_peft_config  # type: ignore
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class HFCausalLMConfig(HFCompatModelConfig):
    neftune_alpha: float | None = None
    generation_config: GenerationConfig | dict | None = None
    peft_config: PeftConfig | dict[str, Any] | None = None # type: ignore
    patcher_config: dict[str, Any] | Literal[False] | None = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        assert self.peft_config is None or PEFT_AVAILABLE, (
            "To use the `peft_config`, you must have PEFT installed."
            " Install it by running `pip install peft`."
        )

        if isinstance(self.peft_config, dict):
            self.peft_config = get_peft_config(self.peft_config)

        if isinstance(self.generation_config, dict):
            self.generation_config = GenerationConfig.from_dict(self.generation_config)

        if torch.cuda.get_device_capability()[0] >= 8:
            self.hf_model_kwargs.setdefault('attn_implementation', 'flash_attention_2')
