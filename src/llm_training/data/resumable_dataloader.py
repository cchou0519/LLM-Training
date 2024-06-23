from typing import Any, Iterator, Protocol, Sized

import torch
import torch.distributed
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              RandomSampler, SequentialSampler)


class ResumableSequentialSampler(SequentialSampler):
    def __init__(self, data_source: Sized) -> None:
        super().__init__(data_source)

        self.used_indices = set()

    def __iter__(self) -> Iterator[int]:
        for i, index in enumerate(super().__iter__()):
            is_used = index in self.used_indices

            self.used_indices.add(index)

            if i + 1 == len(self.data_source):
                self.used_indices.clear()
            
            if not is_used:
                yield index


class ResumableRandomSampler(RandomSampler):
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: torch.Generator | None = None
    ) -> None:
        generator = generator or torch.Generator().manual_seed(0)

        super().__init__(data_source, replacement, num_samples, generator)

        self.used_indices = set()

    def __iter__(self) -> Iterator[int]:
        for i, index in enumerate(super().__iter__()):
            is_used = index in self.used_indices

            self.used_indices.add(index)

            if i + 1 == self.num_samples:
                self.used_indices.clear()
            
            if not is_used:
                yield index


class ResumableDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        self.used_indices = set()

    def __iter__(self) -> Iterator[int]:
        for i, index in enumerate(super().__iter__()):
            is_used = index in self.used_indices
            is_padding = i >= len(self.dataset)

            self.used_indices.add(index)

            if i + 1 == self.num_samples:
                self.used_indices.clear()
            
            if not is_used or is_padding:
                yield index


class ResumableSamplerProtocol(Protocol):
    used_indices: set[int]


class ResumableDataLoader(DataLoader):
    sampler: ResumableSamplerProtocol

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        sampler: None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        generator: torch.Generator | None = None,
        *args,
        **kwargs
    ):
        assert sampler is None, f"To use {self.__class__.__name__}, `sampler` must be None."

        if num_replicas is not None and rank is not None:
            kwargs['sampler'] = ResumableDistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle
            )
        elif shuffle:
            kwargs['sampler'] = ResumableRandomSampler(dataset, generator=generator)
        else:
            kwargs['sampler'] = ResumableSequentialSampler(dataset)

        super().__init__(dataset, *args, **kwargs)

    def sync_state(self) -> None:
        if not torch.distributed.is_initialized():
            return
        
        # sync used_indices
        l = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(l, self.sampler.used_indices)
        self.sampler.used_indices.clear()
        self.sampler.used_indices.update(y for x in l for y in x)

    def state_dict(self) -> dict[str, Any]:
        self.sync_state()

        return {
            'sampler.used_indices': self.sampler.used_indices
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sampler.used_indices = state_dict['sampler.used_indices']
