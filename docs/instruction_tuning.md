# Instruction Tuning

Instruction tuning is implemented by the [`InstructionTuningDataModule`](/src/llm_training/data/instruction_tuning/instruction_tuning_datamodule.py).

Same as `PreTrainingDataModule`, it uses [datasets](https://github.com/huggingface/datasets) under the hood to load and process data.

A valid dataset must include a `messages` field, which is an array.
The format can be referenced as follows:
```json
[
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"}
]
```

See [Templates for Chat Models](https://huggingface.co/docs/transformers/main/en/chat_templating) for more details.

## Key Parameters

| Parameter        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset_kwargs` | kwargs to be passed to [`datasets.load_dataset`](https://huggingface.co/docs/datasets/loading) for loading data.                                                                                                                                                                                                                                                                                                                                                                                                    |
| `tokenizer`      | A transformers tokenizer for tokenizing the data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `chat_template`  | If the value exists in the predefined templates, the predefined template will be selected. Otherwise, the value will be used directly as a Jinja2 syntax template. If the value is None, the tokenizer's built-in template will be used. Note that using the original templates like llama-3, phi-3, etc., directly often leads to incorrect labels. Therefore, it is recommended to use predefined templates. If the desired predefined template does not exist, you should modify the original template yourself. |
| `packing_method` | Methods for concatenating data. `NO_PACKING` will do nothing. `GROUP_BY_LENGTH` will group data based on the length of each entry, ensuring that the combined length of each group does not exceed `max_length`                                                                                                                                                                                                                                                                                                     |
| `max_length`     | Max length of the tokenized data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `num_proc`       | Number of CPU cores for processing data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

For a complete set of parameters, please refer to [`InstructionTuningDataModuleConfig`](/src/llm_training/data/instruction_tuning/instruction_tuning_datamodule_config.py).

## Example

```yaml
...
data:
  class_path: llm_training.data.InstructionTuningDataModule
  init_args.config:
    dataset_kwargs:
      path: ShinoharaHare/Infinity-Instruct-Reformatted
      name: "0625"
    tokenizer:
      class_path: HFTokenizer
      init_args:
        path: microsoft/Phi-3-mini-128k-instruct
    batch_size: 1
    add_default_system_prompt_rate: 0.0
    default_system_prompt: ""
    chat_template: phi-3
    packing_method: GROUP_BY_LENGTH
    max_length: 4096
    pad_to_multiple_of: 64
    validation_split: null
    num_proc: 4
    num_workers: 4
    enable_cache: true
```
