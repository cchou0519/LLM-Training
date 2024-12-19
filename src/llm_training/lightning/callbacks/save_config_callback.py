import os
from pathlib import Path
from typing import Any

import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback as _SaveConfigCallback

from llm_training.lightning.loggers.wandb import WandbLogger


class SaveConfigCallback(_SaveConfigCallback):
    
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            config_path = Path(logger.save_dir, self.config_filename)

            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile
            )

            with open(config_path) as f:
                self.yaml_config = yaml.safe_load(f)

            self.yaml_config['world_size'] = trainer.world_size

            if 'SLURM_JOB_ID' in os.environ:
                self.yaml_config['slurm_job_id'] = os.environ['SLURM_JOB_ID']
                self.yaml_config['slurm_job_name'] = os.environ.get('SLURM_JOB_NAME', None)
                self.yaml_config['slurm_num_nodes'] = os.environ.get('SLURM_NNODES', None)
                self.yaml_config['slurm_ntasks'] = os.environ.get('SLURM_NTASKS', None)

            logger.log_hyperparams(self.yaml_config)
            logger.experiment.save(config_path, policy='now')
            logger.experiment.log_code(include_fn=_wandb_code_include_fn)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]) -> None:
        yaml_config = self.yaml_config if trainer.is_global_zero else None
        checkpoint['config'] = trainer.strategy.broadcast(yaml_config, src=0)


def _wandb_code_include_fn(path: str, root: str):
    p = Path(path).relative_to(root)
    return p.parts[0] in ['src', 'scripts'] and p.suffix in ['.py', '.sh']
