import random
from typing import Any

import torch

from llm_training.data.base_datacollator import BaseDataCollator

from .instruction_tuning_datamodule_config import (
    ConcatMethod, InstructionTuningDataModuleConfig)

PH_PADDING = -1
PH_SEPERATOR = -2


class InstructionTuningDataCollator(BaseDataCollator):
    config: InstructionTuningDataModuleConfig
    
    @property
    def tokenizer(self):
        return self.config.tokenizer

    def __init__(self, config: InstructionTuningDataModuleConfig):
        super().__init__(config)

        assert 'pad_token' in config.tokenizer.special_tokens_map, '`pad_token` is not specified. Please set it manually.'

    def _merge_grouped(self, x: list[list[int]], sequence: list[int]) -> list[int]:
        return [y for i in sequence for y in [*x[i], PH_SEPERATOR]][:-1]

    def _pad_to_longest(self, x: list[list[int]]):
        n = max(len(y) for y in x)
        if self.config.pad_to_multiple_of is not None:
            n = ((n // self.config.pad_to_multiple_of) + 1) * self.config.pad_to_multiple_of
        
        for y in x:
            num_paddings = n - len(y)
            paddings = [PH_PADDING] * num_paddings
            y[:] = paddings + y if self.tokenizer.padding_side == 'left' else y + paddings
        return x

    def __call__(self, batch: list[dict[str, Any]]):
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        
        for x in batch:
            if self.config.concat_method == ConcatMethod.NO_CONCAT:
                input_ids = x['input_ids']
                labels = x['labels']
                position_ids = list(range(len(input_ids)))
            elif self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
                n = len(x['input_ids_group'])
                sequence = random.sample(range(n), k=n)
                input_ids = self._merge_grouped(x['input_ids_group'], sequence)
                labels = self._merge_grouped(x['labels_group'], sequence)

                if self.config.reset_position_ids:
                    position_ids_group = [list(range(len(y))) for y in x['input_ids_group']]
                    position_ids = self._merge_grouped(position_ids_group, sequence)
                    for i, y in enumerate(position_ids):
                        if y == PH_SEPERATOR:
                            position_ids[i] = position_ids[i - 1] + 1
                else:
                    position_ids = list(range(len(input_ids)))

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_position_ids.append(position_ids)

        batch_input_ids = self._pad_to_longest(batch_input_ids)
        batch_labels = self._pad_to_longest(batch_labels)
        batch_position_ids = self._pad_to_longest(batch_position_ids)

        input_ids = torch.tensor(batch_input_ids)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(batch_labels)
        position_ids = torch.tensor(batch_position_ids)

        # replace placeholders
        padding_mask = input_ids == PH_PADDING
        seperator_mask = input_ids == PH_SEPERATOR

        input_ids = input_ids.masked_fill(padding_mask, self.tokenizer.pad_token_id)
        input_ids = input_ids.masked_fill(seperator_mask, self.tokenizer.eos_token_id)
        attention_mask = attention_mask.masked_fill(padding_mask, 0)
        labels = labels.masked_fill(padding_mask | seperator_mask, -100)
        position_ids = position_ids.masked_fill(padding_mask, 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }
