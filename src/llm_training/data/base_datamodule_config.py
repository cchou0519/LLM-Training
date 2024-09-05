from pydantic import BaseModel, ConfigDict


class BaseDataModuleConfig(BaseModel):
    pre_processed_data_path: str | None = None
    validation_split: int | float | None = None
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    prepare_data_per_node: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)
