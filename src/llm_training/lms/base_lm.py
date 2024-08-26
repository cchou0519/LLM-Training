import inspect
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Callable, ContextManager, ParamSpec, TypeVar

import torch
import torch.distributed
import torch.utils.checkpoint
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy, Strategy
from tqdm.auto import tqdm

from llm_training.models.base_model.base_model import BaseModel
from llm_training.utils.context_managers import ContextManagers

from .base_lm_config import BaseLightningModuleConfig

P = ParamSpec('P')
R = TypeVar('R')

logger = logging.getLogger(__name__)

class BaseLightningModule(LightningModule):
    def __init__(self, config: BaseLightningModuleConfig) -> None:
        super().__init__()

        self.config = config
        self._grad_norm = None
        self.configure_model = self._wrap_configure_model(self.configure_model)
        
    @property
    def has_pre_trained_weights(self) -> bool:
        return False

    @property
    def strategy(self) -> Strategy | None:
        return None if self._trainer is None else self.trainer.strategy
    
    @property
    def is_loading_from_checkpoint(self) -> bool:
        if self._trainer is None:
            return False
        
        if self.trainer.ckpt_path is not None:
            return True

        current_frame = inspect.currentframe().f_back
        while current_frame is not None:
            f_locals = current_frame.f_locals
            current_frame = current_frame.f_back

            if isinstance(f_locals.get('self', None), Trainer) and 'ckpt_path' in f_locals:
                return f_locals['ckpt_path'] is not None

    @property
    def should_load_pre_trained_weights(self) -> bool:
        return (
            self.has_pre_trained_weights
            and self.config.load_pre_trained_weights
            and not self.is_loading_from_checkpoint
        )
    
    @property
    def should_initialize_weights(self) -> bool:
        return (
            self.config.init_weights
            and not self.is_loading_from_checkpoint
            and not self.has_pre_trained_weights
        )
    
    @property
    def should_sync_weights(self) -> bool:
        return self.should_load_pre_trained_weights or self.should_initialize_weights

    @property
    def grad_norm(self) -> torch.Tensor | float | None:
        if isinstance(self.strategy, DeepSpeedStrategy):
            return self.strategy.deepspeed_engine.get_global_grad_norm()
        return self._grad_norm

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    @staticmethod
    @contextmanager
    def shutdown_ds_init_context():
        import deepspeed  # type: ignore
        deepspeed.zero.partition_parameters.shutdown_init_context()
        yield
        deepspeed.zero.partition_parameters.restore_init_context()

    def get_pre_trained_weights_context(self) -> ContextManagers:
        context_managers = []
        if isinstance(self.strategy, DeepSpeedStrategy):
            context_managers.append(self.shutdown_ds_init_context())
        return ContextManagers(context_managers)

    def _ds_z3_load_state_dict(self, state_dict: dict[str, torch.Tensor] | None):
        import deepspeed  # type: ignore

        with tqdm(
            desc='Loading weights',
            total=sum(1 for _ in self.parameters()),
            disable=self.global_rank != 0
        ) as progress_bar:
            for n, p in self.named_parameters():
                with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                    if self.global_rank == 0:
                        p.data.copy_(state_dict[n])
                
                progress_bar.set_postfix_str(n)
                progress_bar.update()

    def load_pre_trained_weights(self) -> None:
        with self.get_pre_trained_weights_context():
            state_dict = self.get_pre_trained_weights() if self.global_rank == 0 else None
        
        if getattr(self.strategy, 'zero_stage_3', False):
            self._ds_z3_load_state_dict(state_dict)
        elif self._trainer is not None and self._trainer.num_devices > 1:
            with tqdm(
                desc='Loading weights',
                total=sum(1 for _ in self.parameters()),
                disable=self.global_rank != 0
            ) as progress_bar:
                for n, p in self.named_parameters():
                    if self.global_rank == 0:
                        p.data.copy_(state_dict[n])

                    torch.distributed.broadcast(p.data, src=0)
                    progress_bar.set_postfix_str(n)
                    progress_bar.update()
        else:
            self.load_state_dict(state_dict)

    def configure_model_context(self) -> ContextManager:
        context_managers = [
            BaseModel.init_weights_context(self.should_initialize_weights)
        ]
        
        if self.strategy is not None and not getattr(self.strategy, 'zero_stage_3', False):
            empty_init = self.has_pre_trained_weights or self.is_loading_from_checkpoint
            context = self.strategy.tensor_init_context(empty_init=empty_init)
            context_managers.append(context)
        
        return ContextManagers(context_managers)

    def on_after_configure_model(self) -> None:
        if self.should_load_pre_trained_weights:
            self.load_pre_trained_weights()
    
    def _wrap_configure_model(self, configure_model: Callable[P, R]) -> Callable[P, R]:
        @wraps(configure_model)
        def wrapped_configure_model():
            with self.configure_model_context():
                configure_model()
            
            self.on_after_configure_model()
        
        return wrapped_configure_model
    
    def configure_optimizers(self):
        assert self.config.optim is not None

        optimizer = self.config.optim.optimizer_class(self.parameters(), **self.config.optim.optimizer_kwargs)

        lr_scheduler_class = self.config.optim.lr_scheduler_class
        lr_scheduler_parameters = inspect.signature(lr_scheduler_class).parameters
        lr_scheduler_kwargs = {}
        if 'num_total_steps' in lr_scheduler_parameters:
            lr_scheduler_kwargs['num_total_steps'] = self.trainer.estimated_stepping_batches
        lr_scheduler_kwargs |= self.config.optim.lr_scheduler_kwargs
        lr_scheduler = self.config.optim.lr_scheduler_class(optimizer, **lr_scheduler_kwargs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step'
            }
        }

    def get_model(self) -> BaseModel:
        raise NotImplementedError()
