import random

from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                        HFBasedDataModule)

from .instruction_tuning_datacollator import InstructionTuningDataCollator
from .instruction_tuning_datamodule_config import (
    ConcatMethod, InstructionTuningDataModuleConfig, OverlongHandlingMethod)


class InstructionTuningDataModule(HFBasedDataModule):
    config: InstructionTuningDataModuleConfig
    datacollator_class = InstructionTuningDataCollator

    def __init__(self, config: InstructionTuningDataModuleConfig) -> None:
        super().__init__(config)
 
    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _apply_chat_template_and_tokenize,
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                chat_template=self.config.chat_template,
                add_default_system_prompt_rate=self.config.add_default_system_prompt_rate,
                default_system_prompt=self.config.default_system_prompt
            ),
            batched=True,
            remove_columns=True,
            num_proc=self.config.num_proc,
            desc='Apply chat template and tokenize'
        )

        if self.config.overlong_handling_method == OverlongHandlingMethod.DROP:
            dataset_dict = dataset_dict.filter(
                _drop_overlong,
                input_columns='input_ids',
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Drop overlong'
            )
        elif self.config.overlong_handling_method == OverlongHandlingMethod.TRUNCATE:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _truncate_overlong,
                batched=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Truncate overlong'
            )
        
        if self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _group_by_length,
                batched=True,
                batch_size=10000,
                remove_columns=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Group by length'
            )
    
        return dataset_dict


def _apply_chat_template_and_tokenize(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | None,
    default_system_prompt: str | None,
    add_default_system_prompt_rate: float | None 
):
    new_batch = {
        'input_ids': [],
        'labels': []
    }

    for messages in batch['messages']:
        # Add an empty system prompt randomly if it does not exist.
        has_system_prompt = any(m['role'] == 'system' for m in messages)
        if (
            not has_system_prompt
            and default_system_prompt is not None
            and add_default_system_prompt_rate is not None
            and random.random() < add_default_system_prompt_rate
        ):
            messages.insert(0, {'role': 'system', 'content': default_system_prompt})

    batch_encoding = tokenizer.apply_chat_template(
        batch['messages'],
        chat_template=chat_template,
        return_dict=True,
        return_assistant_tokens_mask=True,
        tokenizer_kwargs=dict(
            return_attention_mask=False,
            verbose=False
        )
    )
    
    for input_ids, assistant_masks in zip(
        batch_encoding['input_ids'],
        batch_encoding['assistant_masks']
    ):
        labels = [i if a == 1 else -100 for i, a in zip(input_ids, assistant_masks)]
        new_batch['input_ids'].append(input_ids)
        new_batch['labels'].append(labels)

    return new_batch


def _drop_overlong(input_ids: list[int], max_length: int):
    return len(input_ids) <= max_length


def _truncate_overlong(batch: dict[str, list], max_length: int):
    for input_ids, labels in zip(batch['input_ids'], batch['labels']):
        if len(input_ids) > max_length:
            input_ids[max_length:] = []
            labels[max_length:] = []
    return batch


def _group_indices_by_length(lengths: list[int], max_length: int) -> list[list[int]]:
    groups = []
    current_group = []
    current_sum = 0
    
    for i, l in sorted(enumerate(lengths), key=lambda x: x[1]):
        if current_sum + l + len(current_group) <= max_length:
            current_group.append(i)
            current_sum += l
        else:
            groups.append(current_group)
            current_group = [i]
            current_sum = l
    
    if current_group:
        groups.append(current_group)
    
    return groups


def _group_by_length(batch: dict[str, list[list[int]]], max_length: int):
    input_ids_group = []
    labels_group = []

    groups = _group_indices_by_length([len(x) for x in batch['input_ids']], max_length)
    for group in groups:
        current_grouped_input_ids = []
        current_grouped_labels = []
        for i in group:
            current_grouped_input_ids.append(batch['input_ids'][i])
            current_grouped_labels.append(batch['labels'][i])
        input_ids_group.append(current_grouped_input_ids)
        labels_group.append(current_grouped_labels)
    
    return {
        'input_ids_group': input_ids_group,
        'labels_group': labels_group
    }
