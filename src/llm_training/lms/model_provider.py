from typing import Any

from jsonargparse.typing import final

from llm_training.models.base_model import BaseModel, BaseModelConfig


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

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'   model_class={self.model_class},\n'
            f'   model_config={repr(self.model_config)}\n'
            ')'
        )
