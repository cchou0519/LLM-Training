import math

import torch
from torch.utils.data import Dataset

from .dummy_datamodule_config import DummyDataModuleConfig


class DummyDataset(Dataset):
    def __init__(self, config: DummyDataModuleConfig) -> None:
        super().__init__()

        self.config = config
        self.base_seed = config.base_seed

        if self.config.num_samples is not None:
            self.num_samples = self.config.num_samples
        elif self.config.num_tokens is not None:
            self.num_samples = math.ceil(self.config.num_tokens / self.config.max_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + index)
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.max_length,), generator=generator)
        return dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids
        )
