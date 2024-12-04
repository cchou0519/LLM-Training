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
            fn_kwargs=dict(
                tokenizer=self.config.tokenizer,
                chat_template=self.config.chat_template
            ),
            batched=True,
            remove_columns=True,
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


def _apply_chat_template_and_tokenize(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | None
):
    new_batch = {
        'chosen_input_ids': [],
        'chosen_labels': [],
        'chosen_length': [],
        'rejected_input_ids': [],
        'rejected_labels': [],
        'rejected_length': []
    }

    chosen_messages = []
    rejected_messages = []
    for prompt, chosen, rejected in zip(
        batch['prompt'],
        batch['chosen'],
        batch['rejected']
    ):
        chosen_messages.append([
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': chosen}
        ])

        rejected_messages.append([
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': rejected}
        ])

    kwargs = dict(
        chat_template=chat_template,
        return_dict=True,
        return_assistant_tokens_mask=True,
        tokenizer_kwargs=dict(
            return_attention_mask=False,
            verbose=False
        )
    )

    chosen_batch_encoding = tokenizer.apply_chat_template(chosen_messages, **kwargs)
    for input_ids, assistant_masks in zip(
        chosen_batch_encoding['input_ids'],
        chosen_batch_encoding['assistant_masks']
    ):
        labels = [i if a == 1 else -100 for i, a in zip(input_ids, assistant_masks)]
        i = input_ids.index(32001)
        assert assistant_masks[i] == 0
        new_batch['chosen_input_ids'].append(input_ids)
        new_batch['chosen_labels'].append(labels)
        new_batch['chosen_length'].append(len(input_ids))

    rejected_batch_encoding = tokenizer.apply_chat_template(rejected_messages, **kwargs)
    for input_ids, assistant_masks in zip(
        rejected_batch_encoding['input_ids'],
        rejected_batch_encoding['assistant_masks']
    ):
        labels = [i if a == 1 else -100 for i, a in zip(input_ids, assistant_masks)]
        new_batch['rejected_input_ids'].append(input_ids)
        new_batch['rejected_labels'].append(labels)
        new_batch['rejected_length'].append(len(input_ids))

    return new_batch


def _drop_overlong(chosen_length: int, rejected_length: int, max_length: int):
    return max(chosen_length, rejected_length) <= max_length


def _truncate_overlong(batch: dict[str, list], max_length: int):
    for prefix in ['chosen', 'rejected']:
        for input_ids, labels in zip(batch[f'{prefix}_input_ids'], batch[f'{prefix}_labels']):
            if len(input_ids) > max_length:
                input_ids[max_length:] = []
                labels[max_length:] = []
    return batch
