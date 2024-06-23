from dataclasses import field

from llm_training.models.lit_base_model import LitBaseConfig


class HFCompatModelConfig(LitBaseConfig):
    hf_path: str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)
    load_hf_weights: bool = True
    trust_remote_code: bool = False
