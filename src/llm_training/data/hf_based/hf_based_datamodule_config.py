from typing import Any

from pydantic import field_validator

from llm_training.data.base_datamodule_config import BaseDataModuleConfig


class HFBasedDataModuleConfig(BaseDataModuleConfig):
    dataset_kwargs: dict[str, Any] | None = None
    num_proc: int | None = None
    cleanup_cache_files: bool = False
    enable_cache: bool = True

    @field_validator('dataset_kwargs')
    def validate_dataset_kwargs(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and 'name' in value:
            value['name'] = str(value['name'])
        return value
