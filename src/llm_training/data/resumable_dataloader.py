from typing import Any, Iterable, Iterator

from lightning import Trainer
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.sampler import Sampler


class ResumableBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
        trainer: Trainer | None = None
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        
        self.trainer = trainer

    def __iter__(self) -> Iterator[list[int]]:
        for i, indices in enumerate(super().__iter__()):
            if self.trainer is not None and i < self.trainer.fit_loop.batch_idx:
                continue

            yield indices


class ResumableDataLoader(DataLoader):
    def __init__(
        self,
        trainer: Trainer | None = None,
        *args,
        **kwargs
    ):
        self.trainer = trainer

        super().__init__(*args, **kwargs)

    def __setattr__(self, attr: str, val: Any):
        if attr == 'batch_sampler':
            assert isinstance(val, BatchSampler)

            val = ResumableBatchSampler(
                sampler=val.sampler,
                batch_size=val.batch_size,
                drop_last=val.drop_last,
                trainer=self.trainer
            )
        
        return super().__setattr__(attr, val)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...
