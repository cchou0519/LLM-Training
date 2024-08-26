from typing import Any

from transformers import PreTrainedTokenizerBase

from llm_training.data.hf_based.hf_based_datamodule import (DatasetDict,
                                                            HFBasedDataModule)

from .preference_tuning_datacollator import PreferenceTuningDataCollator
from .preference_tuning_datamodule_config import (
    OverlongHandlingMethod, PreferenceTuningDataModuleConfig)


class PreferenceTuningDataModule(HFBasedDataModule):
    config: PreferenceTuningDataModuleConfig
    datacollator_class = PreferenceTuningDataCollator

    def __init__(self, config: PreferenceTuningDataModuleConfig) -> None:
        super().__init__(config)
 
    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _apply_chat_template_and_tokenize,
            remove_columns=True,
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                chat_template=self.config.chat_template
            ),
            num_proc=self.config.num_proc,
            desc='Apply chat template and tokenize'
        )

        if self.config.overlong_handling_method == OverlongHandlingMethod.DROP:
            dataset_dict = dataset_dict.filter(
                _drop_overlong,
                input_columns=['chosen_length', 'rejected_length'],
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
    
        return dataset_dict


def _apply_chat_template_and_tokenize_single(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | None
) -> tuple[list[int], list[int]]:
    input_ids = []
    labels = []

    system_prompt = None
    if messages[0]['role'] == 'system':
        system_prompt = messages.pop(0)

    for i, message in enumerate(messages):
        conversation = [message]
        if i == 0 and system_prompt is not None:
            conversation.insert(0, system_prompt)
        text = tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=message['role'] == 'user',
            index=i,
            length=len(messages)
        )
        # 這裡將同一筆資料分多次 tokenize，為保證跟一次 tokenize 全部的結果相同
        # 先在前面加一個 token，encode 後再移除掉
        text = tokenizer.bos_token + text
        current_input_ids = tokenizer.encode(text, add_special_tokens=False)
        current_input_ids = current_input_ids[1:]
        
        if message['role'] in ['system', 'user']:
            input_ids += current_input_ids
            labels += [-100] * len(current_input_ids)
        elif message['role'] == 'assistant':
            input_ids += current_input_ids
            labels += current_input_ids
        else:
            raise ValueError(f"Unknown role: `{message['role']}`")

    return input_ids, labels


def _apply_chat_template_and_tokenize(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | None
):
    chosen_input_ids, chosen_labels = _apply_chat_template_and_tokenize_single(
        [
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': example['chosen']}
        ],
        tokenizer=tokenizer,
        chat_template=chat_template
    )

    rejected_input_ids, rejected_labels = _apply_chat_template_and_tokenize_single(
        [
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': example['rejected']}
        ],
        tokenizer=tokenizer,
        chat_template=chat_template
    )

    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_labels': chosen_labels,
        'chosen_length': len(chosen_input_ids),
        'rejected_input_ids': rejected_input_ids,
        'rejected_labels': rejected_labels,
        'rejected_length': len(rejected_input_ids),
    }


def _drop_overlong(chosen_length: int, rejected_length: int, max_length: int):
    return max(chosen_length, rejected_length) <= max_length


def _truncate_overlong(batch: dict[str, list], max_length: int):
    for prefix in ['chosen', 'rejected']:
        for input_ids, labels in zip(batch[f'{prefix}_input_ids'], batch[f'{prefix}_labels']):
            if len(input_ids) > max_length:
                input_ids[max_length:] = []
                labels[max_length:] = []
    return batch
