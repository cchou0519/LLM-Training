import logging
import sys
from io import StringIO
from pathlib import Path
from typing import TextIO

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class OutputRedirection(Callback):
    LOG_FILE_SUFFIX: str = '.log'

    def __init__(
        self,
        log_file_name: str = '{index}-{version}',
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        enabled: bool = True
    ) -> None:
        super().__init__()

        self.log_file_name = log_file_name
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr
        self.enabled = enabled

        if not enabled:
            return

        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log_file = None
        self.buffer = StringIO()

        if self.redirect_stdout:
            self.stdout_redirector = _StreamRedirector(self.stdout, self.buffer)
            sys.stdout = self.stdout_redirector
        
        if self.redirect_stderr:
            self.stderr_redirector = _StreamRedirector(self.stderr, self.buffer)
            sys.stderr = self.stderr_redirector

        self.redirect_loggers()

    def redirect_loggers(self) -> None:
        logger_names = [None, 'llm_training', 'lightning', 'lightning.fabric', 'lightning.pytorch']
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    original_stream = handler.stream.src if isinstance(handler.stream, _StreamRedirector) else handler.stream
                    if original_stream is self.stdout and self.redirect_stdout:
                        handler.setStream(self.stdout_redirector)
                    elif original_stream is self.stderr and self.redirect_stderr:
                        handler.setStream(self.stderr_redirector)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not self.enabled:
            return
        
        log_file_name = None

        log_dir = Path(trainer.log_dir)
        if trainer.is_global_zero:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_name = self.log_file_name.format(
                index=len(list(log_dir.glob(f'*{self.LOG_FILE_SUFFIX}'))),
                version=trainer.logger.version or trainer.logger.name
            )
            log_file_name += self.LOG_FILE_SUFFIX

        log_file_name = trainer.strategy.broadcast(log_file_name)
        self.log_file = open(log_dir / log_file_name, 'a', encoding='utf-8')
        
        self.log_file.write(self.buffer.getvalue())
        self.buffer.truncate(0)
        self.buffer.seek(0)

        if self.redirect_stdout:
            self.stdout_redirector.dst = self.log_file
        
        if self.redirect_stderr:
            self.stderr_redirector.dst = self.log_file


class _StreamRedirector:
    def __init__(self, src: TextIO, dst: TextIO) -> None:
        self.src = src
        self.dst = dst

    def write(self, s: str) -> int:
        r = self.src.write(s)
        self.dst.write(s)
        return r

    def flush(self) -> None:
        self.src.flush()
        self.dst.flush()
