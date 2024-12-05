from typing import Callable

from llm_training.lms.model_provider import ModelProvider
from llm_training.models.base_model.base_model import BaseModel

ModelType = ModelProvider | BaseModel | Callable[[], BaseModel]


def get_model(model_or_provider: ModelType) -> BaseModel:
    if isinstance(model_or_provider, BaseModel):
        return model_or_provider
    return model_or_provider()
