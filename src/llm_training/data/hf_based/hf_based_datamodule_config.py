from typing import Any

from llm_training.data.base_datamodule_config import BaseDataModuleConfig


class HFBasedDataModuleConfig(BaseDataModuleConfig):
    dataset_kwargs: dict[str, Any] | None = None
    num_proc: int | None = None
    cleanup_cache_files: bool = False
    enable_cache: bool = True

    def __post_init__(self):
        super().__post_init__()

        if 'name' in self.dataset_kwargs:
            self.dataset_kwargs['name'] = str(self.dataset_kwargs['name'])
