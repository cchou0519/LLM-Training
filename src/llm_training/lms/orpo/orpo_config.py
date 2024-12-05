from llm_training.lms.base_lm_config import BaseLightningModuleConfig
from llm_training.lms.utils import ModelType


class ORPOConfig(BaseLightningModuleConfig):
    model: ModelType
    beta: float = 0.1
    ignore_index: int = -100
    empty_cache_threshold: int | None = None
