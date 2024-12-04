from typing import Any

from transformers import PretrainedConfig

from llm_training.models.hf_compat_model import HFCompatModelConfig


class HFCausalLMConfig(HFCompatModelConfig):
    enable_gradient_checkpointing: bool = False
    enable_liger_kernel: bool = False

    hf_config: PretrainedConfig | None = None

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.hf_config, name):
            return getattr(self.hf_config, name)
        return super().__getattr__(name)
