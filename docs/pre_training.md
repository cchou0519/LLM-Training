# Pre-training

The pre-training data processing logic is implemented through [`PreTrainingDataModule`](/src/llm_training/data/pre_training/pre_training_datamodule.py).

It uses [datasets](https://github.com/huggingface/datasets) under the hood to load and process data.

A valid input dataset must include a `text` field, which must be of type string.

## Key Parameters

| Parameter        | Description                                                                                                     |
| :--------------- | :-------------------------------------------------------------------------------------------------------------- |
| `dataset_kwargs` | kwargs to be passed to [`datasets.load_dataset`](https://huggingface.co/docs/datasets/loading) for loading data |
| `tokenizer`      | A transformers tokenizer for tokenizing the data                                                                |
| `max_length`     | Max length of the tokenized data                                                                                |
| `num_proc`       | Number of CPU cores for processing data                                                                         |

For a complete set of parameters, please refer to [`PreTrainingDataModuleConfig`](/src/llm_training/data/pre_training/pre_training_datamodule_config.py).

## Pre-processing data before training

Before training begins, the framework automatically processes the data, ensuring everything is ready before training starts. This is particularly convenient when dealing with small training dataset. However, pre-training datasets are typically large, and CPUs used during training are often limited, making this step very time-consuming.

To address this issue, you can set `pre_processed_data_path` and use many CPUs to execute `scripts/pre_process_pre_training_data.py` for pre-processing and saving the data in advance.

Remember to set `num_proc` to the desired number of CPUs to utilize.

```yaml
data:
  class_path: llm_training.data.PreTrainingDataModule
  init_args.config:
    ...
    pre_processed_data_path: <PATH_TO_SAVE_PRE_PROCESSED_DATA>
    num_proc: <NUMBER_OF_CORES>
```

```bash
python scripts/pre_process_pre_training_data.py -c <CONFIG_PATH>
```

## Example

```yaml
...
data:
  class_path: llm_training.data.PreTrainingDataModule
  init_args.config:
    dataset_kwargs:
      path: HuggingFaceFW/fineweb
      name: sample-10BT
      num_proc: 32 # Number of threads for downloading
    pre_processed_data_path: data/pre_processed/phi-3/fineweb-sample-10bt
    tokenizer: # Phi-3 Tokenizer
      class_path: HFTokenizer
      init_args.path: microsoft/Phi-3-mini-128k-instruct
    batch_size: 1
    max_length: 4096
    validation_split: 30000
    num_proc: 32 # Number of cores for processing
    num_workers: 4 # Number of workers for data loader
```
