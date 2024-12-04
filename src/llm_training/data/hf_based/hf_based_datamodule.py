import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from datasets import DatasetDict as _DatasetDict
from datasets import (Features, disable_caching, enable_caching,
                      is_caching_enabled, load_dataset, load_from_disk)
from datasets.fingerprint import (Hasher, format_kwargs_for_fingerprint,
                                  format_transform_for_fingerprint,
                                  update_fingerprint)
from transformers import PreTrainedTokenizerBase

from llm_training.data.base_datamodule import BaseDataModule

from .hf_based_datamodule_config import HFBasedDataModuleConfig

logger = logging.getLogger(__name__)


class DatasetDict(_DatasetDict, dict[str, Dataset]): ...


class HFBasedDataModule(BaseDataModule):
    config: HFBasedDataModuleConfig
    
    raw_dataset_dict: DatasetDict
    pre_processed_dataset_dict: DatasetDict
    dataset_dict: DatasetDict

    def __init__(self, config: HFBasedDataModuleConfig) -> None:
        super().__init__(config)

    def load_data(self) -> DatasetDict:
        assert self.config.dataset_kwargs is not None

        dataset_kwargs = self.config.dataset_kwargs.copy()
        dataset_kwargs.setdefault('num_proc', self.config.num_proc)
        
        dataset_dict = load_dataset(**dataset_kwargs)

        if isinstance(dataset_dict, Dataset):
            dataset_dict = DatasetDict({'train': dataset_dict})

        assert self.config.validation_split is None or 'train' in dataset_dict and 'validation' not in dataset_dict

        if self.config.cleanup_cache_files:
            n = dataset_dict.cleanup_cache_files()
            logger.info(f'Cleanup cache files: {n}')

        return dataset_dict
    
    def split_data(self, dataset_dict: DatasetDict):
        if self.config.validation_split is not None:
            dataset_dict = dataset_dict['train'].train_test_split(self.config.validation_split, seed=42)
            dataset_dict['validation'] = dataset_dict.pop('test')
        return dataset_dict
    
    def prepare_data(self) -> None:
        if self.config.pre_processed_data_path is None:
            with cache_context(self.config.enable_cache):
                dataset_dict = self.load_data()
                self.pre_process_data(dataset_dict)
    
    def setup(self, stage: str | None = None) -> None:
        with cache_context(self.config.enable_cache):
            super().setup(stage)

    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        return super().pre_process_data(dataset_dict)
    
    def post_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        return super().post_process_data(dataset_dict)

    def save_pre_processed_data(self, path: str | None = None) -> None:
        path = path or self.config.pre_processed_data_path
        self.pre_processed_dataset_dict.save_to_disk(path, num_proc=self.config.num_proc)
    
    def load_pre_processed_data(self, path: str | None = None) -> None:
        path = path or self.config.pre_processed_data_path
        self.pre_processed_dataset_dict = load_from_disk(path)

    def cleanup_cache_files(self) -> None:
        if hasattr(self.raw_dataset_dict, 'cleanup_cache_files'):
            self.raw_dataset_dict.cleanup_cache_files()
    
    @classmethod
    def hash_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> str:
        hasher = Hasher()
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            for p in sorted(Path(tmpdir).glob('*')):
                with open(p, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()

    @classmethod
    def hash_fn_kwargs(cls, fn_kwargs: dict[str, Any]) -> str:
        kwargs = fn_kwargs.copy()
        for k, v in fn_kwargs.items():
            if isinstance(v, PreTrainedTokenizerBase):
                kwargs[k] = cls.hash_tokenizer(v)
        return Hasher.hash(kwargs)
    
    @classmethod
    def map_dataset(
        cls,
        dataset: Dataset,
        function: Callable | None = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: str | list[str] | None = None,
        batched: bool = False,
        batch_size: int | None = 1000,
        drop_last_batch: bool = False,
        remove_columns: str | list[str] | None = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool | None = None,
        cache_file_name: str | None = None,
        writer_batch_size: int | None = 1000,
        features: Features | None = None,
        disable_nullable: bool = False,
        fn_kwargs: dict | None = None,
        num_proc: int | None = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: str | None = None,
        desc: str | None = None
    ):
        dataset_kwargs = {
            'shard': dataset,
            'function': function,
            'with_indices': with_indices,
            'with_rank': with_rank,
            'input_columns': input_columns,
            'batched': batched,
            'batch_size': batch_size,
            'drop_last_batch': drop_last_batch,
            'remove_columns': remove_columns,
            'keep_in_memory': keep_in_memory,
            'writer_batch_size': writer_batch_size,
            'features': features,
            'disable_nullable': disable_nullable,
            'fn_kwargs': cls.hash_fn_kwargs(fn_kwargs) if fn_kwargs is not None else fn_kwargs
        }

        if new_fingerprint is None:
            transform = format_transform_for_fingerprint(Dataset._map_single)
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), dataset_kwargs)
            kwargs_for_fingerprint['fingerprint_name'] = 'new_fingerprint'
            new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)

        dataset = dataset.map(
            function=function,
            with_indices=with_indices,
            with_rank=with_rank,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
            desc=desc,
        )
        
        return dataset

    @classmethod
    def map_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        function: Callable | None = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: str | list[str] | None = None,
        batched: bool = False,
        batch_size: int | None = 1000,
        drop_last_batch: bool = False,
        remove_columns: str | list[str] | bool | None = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool | None = None,
        cache_file_names: dict[str, str | None] | None = None,
        writer_batch_size: int | None = 1000,
        features: Features | None = None,
        disable_nullable: bool = False,
        fn_kwargs: dict | None = None,
        num_proc: int | None = None,
        desc: str | None = None,
    ):
        if cache_file_names is None:
            cache_file_names = {k: None for k in dataset_dict}

        return DatasetDict({
            k: cls.map_dataset(
                dataset,
                function=function,
                with_indices=with_indices,
                with_rank=with_rank,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=dataset.column_names if remove_columns == True else remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[k],
                writer_batch_size=writer_batch_size,
                features=features,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                desc=desc
            ) for k, dataset in dataset_dict.items()
        })


@contextmanager
def cache_context(enabled: bool):
    def set_cache_enabled(b: bool):
        if b:
            enable_caching()
        else:
            disable_caching()
    
    e = is_caching_enabled()
    set_cache_enabled(enabled)

    yield

    set_cache_enabled(e)
