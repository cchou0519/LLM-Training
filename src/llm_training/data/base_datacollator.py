from abc import ABC, abstractmethod
from typing import Any

from .base_datamodule_config import BaseDataModuleConfig


class BaseDataCollator(ABC):
    def __init__(self, config: BaseDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]: ...
