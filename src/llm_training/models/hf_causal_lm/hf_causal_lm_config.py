from llm_training.models.hf_compat_model import HFCompatModelConfig

class HFCausalLMConfig(HFCompatModelConfig):
    enable_gradient_checkpointing: bool = False
