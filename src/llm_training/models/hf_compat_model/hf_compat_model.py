from contextlib import contextmanager
from typing import Generator

import torch
from accelerate import init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from llm_training.models.base_model.base_model import BaseModel

from .hf_compat_config import HFCompatModelConfig


class HFCompatModel(BaseModel):
    config: HFCompatModelConfig
    
    hf_config_class: type[AutoConfig] | type[PretrainedConfig] = AutoConfig
    hf_model_class: type[_BaseAutoModelClass] | type[PreTrainedModel] = AutoModel

    @property
    def has_pre_trained_weights(self) -> bool:
        return (
            super().has_pre_trained_weights
            or (self.config.hf_path is not None and self.config.load_hf_weights)
        )

    def __init__(self, config: HFCompatModelConfig) -> None:
        super().__init__(config)

        if self.config.hf_path is not None:
            self.hf_config = self.load_hf_config()
            self.merge_hf_config(self.hf_config)
    
    def merge_hf_config(self, hf_config: PretrainedConfig) -> None: ...

    def load_hf_config(self, **kwargs) -> PretrainedConfig:
        default_kwargs = {
            'trust_remote_code': self.config.trust_remote_code,
            'revision': self.config.revision,
            'attn_implementation': self.config.attn_implementation
        }
        kwargs = default_kwargs | kwargs
        return self.hf_config_class.from_pretrained(self.config.hf_path, **kwargs)

    def load_hf_model(self, **kwargs) -> "PreTrainedModel":
        default_kwargs = {
            'low_cpu_mem_usage': self.config.low_cpu_mem_usage,
            'torch_dtype': self.config.torch_dtype,
            'trust_remote_code': self.config.trust_remote_code,
            'revision': self.config.revision
        }
        kwargs = default_kwargs | kwargs
        return self.hf_model_class.from_pretrained(self.config.hf_path, **kwargs)

    def load_hf_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        path = self.config.hf_tokenizer_path or self.config.hf_path
        default_kwargs = {
            'trust_remote_code': self.config.trust_remote_code,
            'revision': self.config.revision
        }
        kwargs = default_kwargs | kwargs
        return AutoTokenizer.from_pretrained(path, **kwargs)
    
    @contextmanager
    def torch_dtype_context(self) -> Generator[None, None, None]:
        original_dtype = torch.get_default_dtype()
        torch_dtype = self.config.torch_dtype
        torch_dtype = original_dtype if torch_dtype == 'auto' else torch_dtype
        torch.set_default_dtype(torch_dtype)
        yield
        torch.set_default_dtype(original_dtype)

    def construct_hf_model(self, **kwargs) -> PreTrainedModel:
        with self.torch_dtype_context():
            if issubclass(self.hf_model_class, _BaseAutoModelClass):
                default_kwargs = {}
                default_kwargs['trust_remote_code'] = self.config.trust_remote_code
                default_kwargs['attn_implementation'] = self.config.attn_implementation
                kwargs = default_kwargs | kwargs
                return self.hf_model_class.from_config(self.hf_config, **kwargs)
            return self.hf_model_class(self.hf_config)
    
    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return hf_state_dict

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return state_dict

    def get_hf_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        model = self.load_hf_model()
        state_dict = model.state_dict()
        state_dict = self.convert_state_dict_from_hf(state_dict)
        return state_dict

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        if self.config.hf_path is not None:
            return self.get_hf_pre_trained_weights()
        return super().get_pre_trained_weights()

    def get_hf_model(self) -> PreTrainedModel:
        with init_empty_weights(include_buffers=False):
            hf_model = self.construct_hf_model()
        state_dict = self.convert_state_dict_to_hf(self.state_dict())
        hf_model.load_state_dict(state_dict, assign=True)
        return hf_model
        