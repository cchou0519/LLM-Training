import os
import sys
from typing import TextIO

from llm_training.data import *
from llm_training.lightning.cli import *
from llm_training.models import *


class OutputStreamRedirector:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for stream in self._streams:
            n += stream.write(s)
        return n
    
    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def main():
    cli = LightningCLI(run=False)

    datamodule = cli.datamodule

    assert isinstance(datamodule, HFBasedDataModule)
    
    config = datamodule.config

    pre_processed_data_path = config.pre_processed_data_path

    assert pre_processed_data_path is not None, "`pre_processed_data_path` should not be `None`."

    if not os.path.exists(pre_processed_data_path) or len(os.listdir(pre_processed_data_path)) == 0:
        config.pre_processed_data_path = None
        datamodule.setup()
        datamodule.save_pre_processed_data(pre_processed_data_path)
    else:
        print(f'`pre_processed_data_path="{pre_processed_data_path}"` is not empty, skipping.')
        datamodule.setup()
    
    with open(os.path.join(pre_processed_data_path, 'info.txt'), 'w') as f:
        datamodule.print_dataset_info(file=OutputStreamRedirector(sys.stdout, f))


if __name__ == '__main__':
    main()
