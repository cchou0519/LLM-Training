import gc
import inspect
import logging
import re
from contextlib import contextmanager
from functools import wraps
from typing import Callable, ContextManager, Iterable, ParamSpec, TypeVar

import torch
import torch.distributed
import torch.utils.checkpoint
from accelerate import init_empty_weights
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import Strategy
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy, _Policy, wrap
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm

from llm_training.models.base_model.base_model import BaseModel
from llm_training.overrides.strategies import DeepSpeedStrategy, FSDPStrategy
from llm_training.utils.context_managers import ContextManagers

from .base_lm_config import BaseLightningModuleConfig

P = ParamSpec('P')
R = TypeVar('R')

logger = logging.getLogger(__name__)

class BaseLightningModule(LightningModule):
    _fsdp_module: FSDP | None

    def __init__(self, config: BaseLightningModuleConfig) -> None:
        super().__init__()

        self.config = config
        self._grad_norm = None
        self._fsdp_module = None
        self.configure_model = self._wrap_configure_model(self.configure_model)
        
    @property
    def has_pre_trained_weights(self) -> bool:
        return False

    @property
    def strategy(self) -> Strategy | None:
        return None if self._trainer is None else self.trainer.strategy
    
    @property
    def is_using_ds_z3(self) -> bool:
        return (
            isinstance(self.strategy, DeepSpeedStrategy)
            and self.strategy.zero_stage_3
        )

    @property
    def is_using_fsdp(self) -> bool:
        return isinstance(self.strategy, FSDPStrategy)
    
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
            and self.config.load_weights
            and not self.is_loading_from_checkpoint
        )
    
    @property
    def should_initialize_weights(self) -> bool:
        return (
            self.config.init_weights
            and not self.is_loading_from_checkpoint
            and not self.config.load_weights
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

    def _get_weight_loading_progress_bar(
        self,
        model: nn.Module | None = None,
        name: str | None = None
    ) -> tqdm:
        model = model or self
        desc = 'Loading weights'
        if name is not None:
            desc = f'{desc} ({name})'
        
        disable = self.global_rank != 0
        disable |= not self.should_initialize_weights and not self.should_load_pre_trained_weights

        return tqdm(
            desc=desc,
            total=sum(1 for _ in model.parameters()),
            disable=disable
        )

    def _ds_z3_load_state_dict(self, state_dict: dict[str, torch.Tensor] | None):
        import deepspeed  # type: ignore

        with self._get_weight_loading_progress_bar() as progress_bar:
            for n, p in self.named_parameters():
                with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                    if self.global_rank == 0:
                        p.data.copy_(state_dict[n])
                
                progress_bar.set_postfix_str(n)
                progress_bar.update()

    def load_pre_trained_weights(self) -> None:
        with self.get_pre_trained_weights_context():
            state_dict = self.get_pre_trained_weights() if self.global_rank == 0 else None
        
        if self.is_using_ds_z3:
            self._ds_z3_load_state_dict(state_dict)
        elif self.is_using_fsdp:
            # Do nothing, weight loading happens in model wrapping
            pass
        elif self._trainer is not None and self._trainer.num_devices > 1:
            with self._get_weight_loading_progress_bar() as progress_bar:
                for n, p in self.named_parameters():
                    if self.global_rank == 0:
                        p.data.copy_(state_dict[n])

                    torch.distributed.broadcast(p.data, src=0)
                    progress_bar.set_postfix_str(n)
                    progress_bar.update()
        else:
            self.load_state_dict(state_dict)

    def configure_model_context(self) -> ContextManager:
        if self.is_using_fsdp:
            return init_empty_weights(include_buffers=False)
        return BaseModel.init_weights_context(self.should_initialize_weights)

    def fsdp_wrap_model(
        self,
        model: nn.Module,
        name: str,
        state_dict: dict[str, torch.Tensor] | None,
        modules_to_wrap: Iterable[type[nn.Module] | str] | None = None,
        auto_wrap_policy: _Policy | None = None,
        training: bool = True
    ) -> FSDP:
        assert (
            modules_to_wrap is not None or auto_wrap_policy is not None
            and modules_to_wrap is None or auto_wrap_policy is None
        )

        module_to_name = {m: f'{name}.{n}' for n, m in model.named_modules()}
        progress_bar = self._get_weight_loading_progress_bar(model, name)

        def param_init_fn(module: torch.nn.Module):
            module_name = module_to_name[module]

            if (
                any(t.is_meta for t in module.parameters(recurse=False))
                or any(t.is_meta for t in module.buffers(recurse=False))
            ):
                module.to_empty(device=self.strategy.root_device, recurse=False)
            
            if self.global_rank == 0:
                if self.should_load_pre_trained_weights:
                    for n, p in module.named_parameters(module_name, recurse=False):
                        p.data.copy_(state_dict[n])

                        progress_bar.set_postfix_str(n)
                        progress_bar.update()
                elif self.should_initialize_weights:
                    model._init_weights_impl(module)
                    n = sum(1 for _ in module.parameters(recurse=False))
                    progress_bar.update(n)

        def wrap_policy_fn(module: nn.Module):
            for m in modules_to_wrap:
                if isinstance(m, nn.Module) and m is module:
                    return True
                
                if isinstance(m, str) and m == module.__class__.__name__:
                    return True
                
                if isinstance(m, type) and m == module.__class__:
                    return True
            
            return False

        if auto_wrap_policy is None:
            auto_wrap_policy = CustomPolicy(wrap_policy_fn)

        model = wrap(
            model,
            param_init_fn=param_init_fn,
            sync_module_states=self.should_sync_weights,
            auto_wrap_policy=auto_wrap_policy
        )

        if training:
            assert self._fsdp_module is None or self.trainer.gradient_clip_val is not None
            self._fsdp_module = model

        progress_bar.close()
        return model

    def on_fsdp_wrap_model(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        raise NotImplementedError()

    def on_after_configure_model(self) -> None:
        if self.config.frozen_modules is not None:
            for n, m in self.named_modules():
                for fm in self.config.frozen_modules:
                    if re.search(rf'{fm}', n) is not None:
                        if self.global_rank == 0:
                            logger.info(f'Freeze `{n}`')
                        m.eval().requires_grad_(False)
                        break

        if self.should_load_pre_trained_weights:
            self.load_pre_trained_weights()

        if self.is_using_fsdp:
            state_dict = None
            if (
                self.global_rank == 0
                and self.should_load_pre_trained_weights
            ):
                with self.get_pre_trained_weights_context():
                    state_dict = self.get_pre_trained_weights()

            self.on_fsdp_wrap_model(state_dict)
            
            del state_dict
            gc.collect()

            self.configure_gradient_clipping = self._fsdp_configure_gradient_clipping
    
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

    def _fsdp_configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
        self._grad_norm = self._fsdp_module.clip_grad_norm_(gradient_clip_val).item()

    def get_model(self) -> BaseModel:
        raise NotImplementedError()
