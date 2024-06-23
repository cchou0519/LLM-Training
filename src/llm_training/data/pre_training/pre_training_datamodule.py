import logging
import math
import os
import pickle
import random
from typing import Any, Iterable

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                        HFBasedDataModule)

from .pre_training_datacollator import PreTrainingDataCollator
from .pre_training_datamodule_config import (ConcatMethod,
                                             PreTrainingDataModuleConfig)

SOURCE_INDEX_FILE_NAME = 'source.idx'

logger = logging.getLogger(__name__)


class PreTrainingDataModule(HFBasedDataModule):
    config: PreTrainingDataModuleConfig
    datacollator_class = PreTrainingDataCollator

    def __init__(self, config: PreTrainingDataModuleConfig) -> None:
        super().__init__(config)

    def _tokenize(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Tokenize')

        if self.config.shuffle_before_tokenization:
            dataset_dict = dataset_dict.shuffle(seed=42)
            dataset_dict = dataset_dict.flatten_indices(num_proc=self.config.num_proc)
        
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _tokenize,
            batched=True,
            remove_columns=True,
            fn_kwargs=dict(tokenizer=self.config.tokenizer),
            num_proc=self.config.num_proc,
            desc='Tokenize'
        )
        return dataset_dict
    
    def _truncate(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Truncate')

        dataset_dict = dataset_dict.map(
            _truncate,
            batched=True,
            fn_kwargs=dict(
                max_length=self.config.max_length,
                stride=self.config.stride
            ),
            num_proc=self.config.num_proc,
            desc='Truncate'
        )
        return dataset_dict
    
    def _concat_and_truncate(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Concat and truncate')

        dataset_dict = dataset_dict.sort('source')
        dataset_dict = dataset_dict.map(
            _concat_and_truncate,
            batched=True,
            batch_size=self.config.concat_and_truncate_batch_size,
            fn_kwargs=dict(max_length=self.config.max_length),
            num_proc=self.config.num_proc,
            desc='Concat and truncate'
        )
        return dataset_dict

    def compute_source_indices(self, dataset: Dataset) -> dict[str, list[int]]:
        source_to_indices: dict[str, list[int]] = {}
        progress = tqdm(total=len(dataset), desc='Partition by source')
        i = 0
        for batch in dataset.select_columns('source').iter(1000):
            for source in batch['source']:
                indices = source_to_indices.setdefault(source, [])
                indices.append(i)
                i += 1
                progress.update()
        return source_to_indices

    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self._tokenize(dataset_dict)

        if self.config.max_length is not None:
            dataset_dict = self._truncate(dataset_dict)

        if self.config.concat_method == ConcatMethod.CONCAT_AND_TRUNCATE:
            dataset_dict = self._concat_and_truncate(dataset_dict)
        
        for k, v in dataset_dict.items():
            if k == 'train':
                self.source_indices = self.compute_source_indices(v)

        return dataset_dict

    def save_pre_processed_data(self, path: str | None = None) -> None:
        path = path or self.config.pre_processed_data_path

        super().save_pre_processed_data(path)
        with open(os.path.join(path, 'source.idx'), 'wb') as f:
            pickle.dump(self.source_indices, f)

    def load_pre_processed_data(self, path: str | None = None) -> DatasetDict:
        path = path or self.config.pre_processed_data_path
        
        super().load_pre_processed_data(path)
        with open(os.path.join(path, SOURCE_INDEX_FILE_NAME), 'rb') as f:
            self.source_indices = pickle.load(f)
    
    def sample_data(
        self,
        dataset_dict: DatasetDict,
        source_indices: dict[str, list[int]],
        seed: int = 42
    ) -> DatasetDict:
        if all(x == 1.0 for x in self.config.sample_rate.values()):
            return dataset_dict
                
        r = random.Random(seed)
        unused_sample_rate = self.config.sample_rate.copy()
        for k, dataset in dataset_dict.items():
            if k == 'train':
                sampled_indices = []
                for source, indices in source_indices.items():
                    sample_rate = self.config.sample_rate.get(source, 1.0)
                    unused_sample_rate.pop(source, None)
                    decimal, integer = math.modf(sample_rate)
                    sampled_indices += indices * int(integer)
                    if decimal > 0.0:
                        n = len(indices)
                        sampled_indices += r.sample(indices, k=int(n * decimal))
                dataset_dict[k] = dataset.select(sampled_indices)
        
        if len(unused_sample_rate) > 0:
            logger.warn(f'Some sources specified by `sample_rate` are not found in the dataset:\n {unused_sample_rate}')
        
        return dataset_dict
    
    def post_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = super().post_process_data(dataset_dict)

        source_indices = self.source_indices
        indices_mapping = dataset_dict['train']._indices
        if indices_mapping is not None:
            indices_mapping = indices_mapping['indices']
            reverse_mapping = {x.as_py(): i for i, x in enumerate(indices_mapping)}
            source_indices = {k: [reverse_mapping[i] for i in v if i in reverse_mapping] for k, v in source_indices.items()}

        dataset_dict = self.sample_data(dataset_dict, source_indices)
        return dataset_dict


def _tokenize(
    batch: dict[str, list[str | Any]],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    keep_columns: bool | Iterable[str] = False
):
    if 'source' not in batch:
        batch['source'] = [None] * len(batch_text)
    
    if keep_columns == True:
        keep_columns = set(batch.keys())
    elif keep_columns == False:
        keep_columns = set()
    else:
        keep_columns = set(keep_columns)
    
    keep_columns.add('source')

    selected_indices = [i for i, x in enumerate(batch['text']) if x]
    batch_text = [batch['text'][i] for i in selected_indices]
    batch = {k: [batch[k][i] for i in selected_indices] for k in keep_columns}   

    batch['input_ids'] = tokenizer(
        batch_text,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=False
    )['input_ids']

    if 'text' in keep_columns:
        batch['text'] = batch_text

    batch['length'] = []
    for input_ids in batch['input_ids']:
        input_ids.insert(0, tokenizer.bos_token_id)
        input_ids.append(tokenizer.eos_token_id)
        batch['length'].append(len(input_ids))
    
    return batch


def _truncate(
    batch: dict[str, list[str | int | Any]],
    max_length: int,
    stride: int
):
    new_batch = {}
    
    for i in range(len(batch['input_ids'])):
        input_ids = batch['input_ids'][i]
        for j in range(0, len(input_ids), stride):
            t = input_ids[j:j + max_length]
            for k in batch.keys():
                l = new_batch.setdefault(k, [])
                if k == 'input_ids':
                    l.append(t)
                elif k == 'length':
                    l.append(len(t))
                else:
                    l.append(batch[k][i])

    return new_batch


def _concat_and_truncate(
    batch: dict[str, list[str | int]],
    max_length: int
):    
    batch_input_ids = []
    batch_source = []
    batch_length = []

    current_source = batch['source'][0]
    current_input_ids = []
    for source, input_ids in zip(batch['source'], batch['input_ids']):
        if len(input_ids) == max_length:
            batch_input_ids.append(input_ids)
            batch_source.append(source)
            batch_length.append(len(input_ids))
            continue

        if source != current_source:
            if current_input_ids:
                batch_input_ids.append(current_input_ids)
                batch_source.append(current_source)
                batch_length.append(len(current_input_ids))
                current_input_ids = []
            current_source = source

        current_input_ids += input_ids
        while len(current_input_ids) >= max_length:
            batch_input_ids.append(current_input_ids[:max_length])
            batch_source.append(current_source)
            batch_length.append(len(current_input_ids[:max_length]))
            current_input_ids[:max_length] = []
    
    if current_input_ids:
        batch_input_ids.append(current_input_ids)
        batch_source.append(current_source)
        batch_length.append(len(current_input_ids))

    return {
        'input_ids': batch_input_ids,
        'source': batch_source,
        'length': batch_length
    }
