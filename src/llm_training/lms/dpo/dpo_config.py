from llm_training.lms.base_lm_config import BaseLightningModuleConfig
from llm_training.lms.utils import ModelType


class DPOConfig(BaseLightningModuleConfig):
    model: ModelType
    ref_model: ModelType | None = None
    beta: float = 0.1
    label_smoothing: float = 0.0
    ignore_index: int = -100
