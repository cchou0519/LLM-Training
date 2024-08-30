from llm_training.data.base_datamodule_config import BaseDataModuleConfig


class HFBasedDataModuleConfig(BaseDataModuleConfig):
    dataset_kwargs: dict | None = None
    num_proc: int | None = None
    cleanup_cache_files: bool = False
    enable_cache: bool = True
