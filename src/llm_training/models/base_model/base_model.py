from contextlib import contextmanager
from typing import ClassVar, Generator

import safetensors.torch
import torch
from torch import nn
from torch.distributed._composable.fsdp import (MixedPrecisionPolicy,
                                                OffloadPolicy)
from torch.distributed.device_mesh import DeviceMesh

from .base_model_config import BaseModelConfig


class BaseModel(nn.Module):
    _init_weights: ClassVar[bool] = True

    config_class: type[BaseModelConfig] = BaseModelConfig
    no_split_modules: list[str] = []

    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()

        self.config = config

        if self._init_weights:
            self.init_weights()
    
    @property
    def has_pre_trained_weights(self) -> bool:
        return self.config.pre_trained_weights is not None

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        return safetensors.torch.load_file(self.config.pre_trained_weights)

    def _init_weights_impl(self, module: nn.Module) -> None: ...

    def init_weights(self) -> None:
        self.apply(self._init_weights_impl)

    @classmethod
    @contextmanager
    def init_weights_context(cls, init_weights: bool) -> Generator[None, None, None]:
        v = cls._init_weights
        cls._init_weights = bool(init_weights)
        yield
        cls._init_weights = v

    def configure_tensor_parallel(self, tp_mesh: DeviceMesh) -> None:
        if tp_mesh.size() == 1:
            return
        
        raise NotImplementedError(f"`{self.__class__.__name__}` does not support tensor parallel.")

    def configure_fully_sharded_data_parallel(
        self,
        dp_mesh: DeviceMesh,
        reshard_after_forward: bool | int,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
        **kwargs
    ) -> None:
        if dp_mesh.size() == 1:
            return
        
        raise NotImplementedError(f"`{self.__class__.__name__}` does not support fully sharded data parallel.")

    def parallelize(
        self,
        dp_mesh: DeviceMesh,
        tp_mesh: DeviceMesh,
        **fsdp_kwargs
    ) -> None:
        self.configure_tensor_parallel(tp_mesh)
        self.configure_fully_sharded_data_parallel(dp_mesh, **fsdp_kwargs)
