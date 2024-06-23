import os
from collections import Counter

from datasets import DatasetDict
from tabulate import tabulate
from tqdm.auto import tqdm

from llm_training.data import *
from llm_training.models import *
from llm_training.overrides.cli import *


def get_tokens_table(dataset_dict: DatasetDict) -> str:
    tokens: dict[str, Counter[str, int]] = {}
    for k, dataset in dataset_dict.items():
        counter = Counter()
        dataset = dataset.select_columns(['source', 'length'])
        progress = tqdm(total=len(dataset), desc=f'Count tokens ({k})')
        for batch in dataset.iter(1000):
            batch_size = len(batch['length'])
            for source, length in zip(batch['source'], batch['length']):
                counter[source] += length
                counter['all'] += length
            tokens[k] = counter
            progress.set_postfix(tokens=counter['all'])
            progress.update(batch_size)
    progress.clear()
    
    return tabulate(
        [
            [split, source, tokens] 
            for split, counter in tokens.items()
            for source, tokens in counter.most_common()
        ],
        headers=['Split', 'Source', 'Tokens'],
        tablefmt='orgtbl'
    )


def main():
    cli = LightningCLI(run=False)

    datamodule: PreTrainingDataModule = cli.datamodule
    config = datamodule.config

    enable_cache = config.enable_cache
    pre_processed_data_path = config.pre_processed_data_path

    if not os.path.exists(pre_processed_data_path) or len(os.listdir(pre_processed_data_path)) == 0:
        config.pre_processed_data_path = None
        config.enable_cache = True
        datamodule.setup()
        datamodule.save_pre_processed_data(pre_processed_data_path)
    else:
        print(f'`pre_processed_data_path="{pre_processed_data_path}"` is not empty, skipping.')
        datamodule.setup()

    if not enable_cache:
        datamodule.cleanup_cache_files()

    table_string = (
        'Original Tokens:\n'
        + get_tokens_table(datamodule.pre_processed_dataset_dict)
        + '\n\n'
        + 'Sampled Tokens:\n'
        + get_tokens_table(datamodule.dataset_dict)
    )

    with open(os.path.join(pre_processed_data_path, 'tokens.txt'), 'w') as f:
        f.write(table_string + '\n')

    print(table_string)


if __name__ == '__main__':
    main()
