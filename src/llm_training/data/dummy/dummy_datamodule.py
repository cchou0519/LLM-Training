from llm_training.data.base_datamodule import BaseDataModule, DatasetDict

from .dummy_datamodule_config import DummyDataModuleConfig
from .dummy_dataset import DummyDataset


class DummyDataModule(BaseDataModule):
    config: DummyDataModuleConfig

    def __init__(self, config: DummyDataModuleConfig) -> None:
        super().__init__(config)

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            self.config.base_seed = self.trainer.strategy.broadcast(self.config.base_seed)
        
        super().setup(stage)

    def load_data(self) -> DatasetDict:
        return {'train': DummyDataset(self.config)}
