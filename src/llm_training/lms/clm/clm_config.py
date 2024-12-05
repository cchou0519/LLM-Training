from llm_training.lms.base_lm_config import BaseLightningModuleConfig
from llm_training.lms.utils import ModelType


class CLMConfig(BaseLightningModuleConfig):
    model: ModelType
    ignore_index: int = -100
    neftune_alpha: float | None = None
    log_perplexity: bool = True
