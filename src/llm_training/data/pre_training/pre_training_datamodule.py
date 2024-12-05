import logging
import math
import random
import shutil
from functools import partial
from typing import Any

import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from tabulate import tabulate
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                            HFBasedDataModule)

from .pre_training_datacollator import PreTrainingDataCollator
from .pre_training_datamodule_config import (PackingMethod,
                                             PreTrainingDataModuleConfig)

logger = logging.getLogger(__name__)


class PreTrainingDataModule(HFBasedDataModule):
    config: PreTrainingDataModuleConfig
    datacollator_class = PreTrainingDataCollator

    def __init__(self, config: PreTrainingDataModuleConfig) -> None:
        super().__init__(config)
    
    @classmethod
    def _tokenize(
        cls,
        batch: dict[str, list[str | Any]],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> dict[str, list]:
        new_batch = {
            'source': [],
            'input_ids': [],
            'length': []
        }

        selected_indices = [i for i, x in enumerate(batch['text']) if x]
        batch['text'] = [batch['text'][i] for i in selected_indices]
        new_batch['source'] = [batch['source'][i] for i in selected_indices]

        new_batch['input_ids'] = tokenizer.batch_encode_plus(
            batch['text'],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            verbose=False
        )['input_ids']
        
        for input_ids in new_batch['input_ids']:
            input_ids.insert(0, tokenizer.bos_token_id)
            input_ids.append(tokenizer.eos_token_id)
            new_batch['length'].append(len(input_ids))
        
        return new_batch

    @classmethod
    def _truncate(
        cls,
        batch: dict[str, list],
        max_length: int,
        stride: int
    ) -> dict[str, list]:
        new_batch = {
            'source': [],
            'input_ids': [],
            'length': []
        }
        
        for i in range(len(batch['input_ids'])):
            source = batch['source'][i]
            input_ids = batch['input_ids'][i]
            for j in range(0, len(input_ids), stride):
                truncated_input_ids = input_ids[j:j + max_length]
                new_batch['source'].append(source)
                new_batch['input_ids'].append(truncated_input_ids)
                new_batch['length'].append(len(truncated_input_ids))

        return new_batch

    @classmethod
    def _naive_packing(
        cls,
        batch: dict[str, list],
        max_length: int
    ) -> dict[str, list]:
        new_batch = {
            'source': [],
            'input_ids': [],
            'attention_mask': [],
            'length': []
        }

        current_source = batch['source'][0]
        current_input_ids = []
        current_attention_mask = []
        for source, input_ids in zip(batch['source'], batch['input_ids']):
            if len(input_ids) == max_length:
                new_batch['source'].append(source)
                new_batch['input_ids'].append(input_ids)
                new_batch['attention_mask'].append([1] * max_length)
                new_batch['length'].append(max_length)
                continue

            if source != current_source:
                if current_input_ids:
                    new_batch['source'].append(current_source)
                    new_batch['input_ids'].append(current_input_ids)
                    new_batch['attention_mask'].append(current_attention_mask)
                    new_batch['length'].append(len(current_input_ids))
                    current_input_ids = []
                    current_attention_mask = []
                current_source = source

            current_input_ids += input_ids
            current_attention_mask += [current_attention_mask[-1] + 1 if current_attention_mask else 1] * len(input_ids)
            while len(current_input_ids) >= max_length:
                new_batch['source'].append(current_source)
                new_batch['input_ids'].append(current_input_ids[:max_length])
                new_batch['attention_mask'].append(current_attention_mask[:max_length])
                new_batch['length'].append(len(current_input_ids[:max_length]))
                current_input_ids[:max_length] = []
                current_attention_mask[:max_length] = []
        
        if current_input_ids:
            new_batch['source'].append(current_source)
            new_batch['input_ids'].append(current_input_ids)
            new_batch['attention_mask'].append(current_attention_mask)
            new_batch['length'].append(len(current_input_ids))

        for attention_mask in new_batch['attention_mask']:
            if attention_mask[0] == 1:
                continue

            offset = attention_mask[0] - 1
            attention_mask[:] = [a - offset for a in attention_mask]

        return new_batch

    @classmethod
    def _split_batch_by_source(cls, batch: dict[str, list]) -> dict[str, dict[str, list]]:
        mapping = {}
        keys = batch.keys()
        first_key = next(iter(keys))
        for i in range(len(batch[first_key])):
            sub_batch = mapping.setdefault(batch['source'][i], {})
            for k in keys:
                l = sub_batch.setdefault(k, [])
                l.append(batch[k][i])
        return mapping

    @classmethod
    def _best_fit_bin_packing(
        cls,
        capacity: int,
        lengths: list[int]
    ) -> list[list[int]]:
        bins = []
        contents = []
        for i, length in enumerate(lengths):
            best_bin_index = -1
            min_space_left = float('inf')

            for j in range(len(bins)):
                if bins[j] >= length and bins[j] - length < min_space_left:
                    best_bin_index = j
                    min_space_left = bins[j] - length

            if best_bin_index != -1:
                bins[best_bin_index] -= length
                contents[best_bin_index].append(i)
            else:
                bins.append(capacity - length)
                contents.append([i])
        return contents

    @classmethod
    def _best_fit_decreasing(
        cls,
        batch: dict[str, list],
        max_length: int
    ) -> dict[str, list]:
        new_batch = {
            'input_ids': [],
            'attention_mask': [],
            'source': [],
            'length': []
        }

        batches = cls._split_batch_by_source(batch)
        for source, sub_batch in batches.items():
            indices = sorted(range(len(sub_batch['length'])), key=lambda i: sub_batch['length'][i], reverse=True)
            sub_batch = {k: [sub_batch[k][i] for i in indices] for k in sub_batch.keys()}
            groups = cls._best_fit_bin_packing(max_length, sub_batch['length'])
            for group in groups:
                current_input_ids = []
                current_attention_mask = []
                for document_idx, example_idx in enumerate(group, start=1):
                    input_ids = sub_batch['input_ids'][example_idx]
                    current_input_ids += input_ids
                    current_attention_mask += [document_idx] * len(input_ids)
                new_batch['input_ids'].append(current_input_ids)
                new_batch['attention_mask'].append(current_attention_mask)
                new_batch['source'].append(source)
                new_batch['length'].append(len(current_input_ids))

        return new_batch
    
    @classmethod
    def _pre_process_data(
        cls,
        batch: dict[str, list],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int | None,
        stride: int | None,
        packing_method: PackingMethod,
    ) -> dict[str, list]:
        batch = cls._tokenize(batch, tokenizer)

        if max_length is not None:
            batch = cls._truncate(batch, max_length, stride)

        if packing_method == PackingMethod.NAIVE_PACKING:
            batch = cls._naive_packing(batch, max_length)
        elif packing_method == PackingMethod.BEST_FIT_BIN_PACKING:
            batch = cls._best_fit_decreasing(batch, max_length)
        
        return batch

    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Sorting data by source')
        for k, v in dataset_dict.items():
            if 'source' in v.column_names:
                dataset_dict[k] = v.sort('source')
            else:
                dataset_dict[k] = v.add_column('source', [None] * len(v))

        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            self._pre_process_data,
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                max_length=self.config.max_length,
                stride=self.config.stride,
                packing_method=self.config.packing_method
            ),
            batched=True,
            batch_size=self.config.pre_processing_batch_size,
            remove_columns=True,
            num_proc=self.config.num_proc,
            features=Features({
                'source': Value('string'),
                'input_ids': Sequence(Value('uint32')),
                'attention_mask': Sequence(Value('uint16')),
                'length': Value('uint32')
            }),
            desc='Pre-processing data'
        )

        return dataset_dict
    
    def _compute_source_indices(self, dataset: Dataset) -> dict[str, list[int]]:
        sources = dataset.data['source'].to_pandas()
        if (indices := dataset._indices) is not None:
            indices = indices['indices'].to_pandas()
            sources = sources.iloc[indices]
        df = pd.DataFrame({
            'source': sources,
            'index': range(len(sources))
        })
        source_indices = df.groupby('source')['index'].apply(list).to_dict()
        return source_indices

    def sample_data(self, dataset: Dataset) -> Dataset:
        sample_rate = self.config.sample_rate

        if all(x == 1.0 for x in sample_rate.values()):
            return dataset
        
        source_indices = self._compute_source_indices(dataset)
        
        r = random.Random(42)
        unused_sample_rate = sample_rate.copy()
        sampled_indices = []
        for source, indices in source_indices.items():
            sr = sample_rate.get(source, 1.0)
            unused_sample_rate.pop(source, None)
            decimal, integer = math.modf(sr)
            sampled_indices += indices * int(integer)
            if decimal > 0.0:
                n = len(indices)
                sampled_indices += r.sample(indices, k=int(n * decimal))
        dataset = dataset.select(sampled_indices)
        
        if len(unused_sample_rate) > 0:
            logger.warning(f'Some sources specified by `sample_rate` are not found in the dataset:\n {unused_sample_rate}')
        
        return dataset

    def post_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = super().post_process_data(dataset_dict)
        if 'train' in dataset_dict:
            logger.info('Sampling data')
            dataset_dict['train'] = self.sample_data(dataset_dict['train'])
            logger.info('Done')
        return dataset_dict
    
    @staticmethod
    def _get_tokens_table(dataset_dict: DatasetDict) -> str:
        tabular_data = {
            'Split': [],
            'Source': [],
            'Tokens': []
        }
        for k, dataset in dataset_dict.items():
            df: pd.DataFrame = dataset.data.select(['source', 'length']).to_pandas()

            if (indices := dataset._indices) is not None:
                indices = indices['indices'].to_pandas()
                df = df.iloc[indices]

            tabular_data['Split'].append(k)
            tabular_data['Source'].append('*')
            tabular_data['Tokens'].append(int(df['length'].sum()))

            df = df.groupby('source', as_index=False)[['length']].sum()
            for r in df.itertuples():
                tabular_data['Split'].append(k)
                tabular_data['Source'].append(r.source)
                tabular_data['Tokens'].append(r.length)

        tabular_data = pd.DataFrame(tabular_data)
        tabular_data.sort_values(['Split', 'Source'], inplace=True)

        return tabulate(
            tabular_data,
            headers='keys',
            tablefmt='orgtbl',
            showindex=False
        )

    def print_dataset_info(self, file: str | None) -> None:
        super().print_dataset_info(file)
        print_ = partial(print, file=file)
        print_('―' * shutil.get_terminal_size().columns, end='\n\n')
        
        logger.info('Counting Original Tokens')
        original_token_table = self._get_tokens_table(self.pre_processed_dataset_dict)

        logger.info('Counting Sampled Tokens')
        sampled_token_table = self._get_tokens_table(self.dataset_dict)

        print_('Original Tokens:', end='\n\n')
        print_(original_token_table, end='\n\n')
        print_('Sampled Tokens:', end='\n\n')
        print_(sampled_token_table)
