from typing import Any
from llm_training.models.base_model import BaseModel, BaseModelConfig
from jsonargparse.typing import final

@final
class ModelProvider:
    def __init__(
        self,
        model_class: type[BaseModel],
        model_config: dict[str, Any] | BaseModelConfig
    ) -> None:
        self.model_class = model_class
        if isinstance(model_config, dict):
            self.model_config = model_class.config_class.model_validate(model_config)
        else:
            self.model_config = model_config

    def __call__(self) -> BaseModel:
        return self.model_class(self.model_config)
