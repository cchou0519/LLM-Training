import inspect
import logging
from contextlib import nullcontext
from functools import wraps
from typing import Callable, ContextManager, ParamSpec, TypeVar

import safetensors
import safetensors.torch
import torch
import torch.distributed
import torch.utils.checkpoint
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy, Strategy
from tqdm.auto import tqdm

from .lit_base_config import LitBaseConfig


P = ParamSpec('P')
R = TypeVar('R')

logger = logging.getLogger(__name__)

class LitBaseModel(LightningModule):
    def __init__(self, config: LitBaseConfig) -> None:
        super().__init__()

        self.config = config
        self._grad_norm = None

        self.configure_model = self._wrap_configure_model(self.configure_model)
        
    @property
    def has_pre_trained_weights(self) -> bool:
        return self.config.pre_trained_weights is not None

    @property
    def strategy(self) -> Strategy | None:
        return None if self._trainer is None else self.trainer.strategy
    
    @property
    def is_load_from_checkpoint(self) -> bool:
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
    def need_to_load_pre_trained_weights(self) -> bool:
        return (
            self.has_pre_trained_weights
            and not self.is_load_from_checkpoint
        )
    
    @property
    def need_to_initialize_weights(self) -> bool:
        return (
            self.config.initialize_weights
            and not self.is_load_from_checkpoint
            and not self.has_pre_trained_weights
        )
    
    @property
    def need_to_sync_weights(self) -> bool:
        return self.need_to_load_pre_trained_weights or self.need_to_initialize_weights

    @property
    def grad_norm(self) -> torch.Tensor | float | None:
        if isinstance(self.strategy, DeepSpeedStrategy):
            return self.strategy.deepspeed_engine.get_global_grad_norm()
        return self._grad_norm

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        return safetensors.torch.load_file(self.config.pre_trained_weights)

    def _ds_z3_load_state_dict(self, state_dict: dict[str, torch.Tensor] | None):
        import deepspeed # type: ignore

        if self.global_rank == 0:
            progress = tqdm(total=sum(1 for _ in self.parameters()), desc='Loading weights')

        for n, p in self.named_parameters():
            with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                if self.global_rank == 0:
                    p.data.copy_(state_dict[n])
                    progress.set_postfix_str(n)
                    progress.update()

        if self.global_rank == 0:
            progress.close()

    def load_pre_trained_weights(self) -> None:
        state_dict = self.get_pre_trained_weights() if self.global_rank == 0 else None
        
        if getattr(self.strategy, 'zero_stage_3', False):
            self._ds_z3_load_state_dict(state_dict)
        elif self._trainer is not None and self._trainer.num_devices > 1:
            if self.global_rank == 0:
                progress = tqdm(total=sum(1 for _ in self.parameters()), desc='Loading weights')

            for n, p in self.named_parameters():
                if self.global_rank == 0:
                    p.data.copy_(state_dict[n])

                    progress.set_postfix_str(n)
                    progress.update()

                torch.distributed.broadcast(p.data, src=0)
        
            if self.global_rank == 0:
                progress.close()
        else:
            self.load_state_dict(state_dict)

    def configure_model_context(self) -> ContextManager:
        context = nullcontext()
        if self.strategy is not None and not getattr(self.strategy, 'zero_stage_3', False):
            empty_init = self.has_pre_trained_weights or self.is_load_from_checkpoint
            context = self.strategy.tensor_init_context(empty_init=empty_init)
        return context

    def on_after_configure_model(self) -> None:
        if self.need_to_load_pre_trained_weights:
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
