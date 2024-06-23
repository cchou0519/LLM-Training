from contextlib import contextmanager, nullcontext

import torch
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from llm_training.models.lit_base_model import LitBaseModel
from llm_training.overrides.strategies.deepspeed import DeepSpeedStrategy

from .hf_compat_config import HFCompatModelConfig


class HFCompatModel(LitBaseModel):
    config: HFCompatModelConfig
    hf_config_class: type[AutoConfig] | type[PretrainedConfig] = AutoConfig
    hf_model_class: type[_BaseAutoModelClass] | type[PreTrainedModel] = AutoModel

    def __init__(self, config: HFCompatModelConfig) -> None:
        super().__init__(config)

        if self.config.hf_path is not None:
            self.hf_config = self.load_hf_config()
            self.update_config_with_hf_config(self.hf_config)
    
    @property
    def has_pre_trained_weights(self) -> bool:
        return super().has_pre_trained_weights or (self.config.hf_path is not None and self.config.load_hf_weights)

    def update_config_with_hf_config(self, hf_config: PretrainedConfig) -> None: ...

    def load_hf_config(self, **kwargs) -> PretrainedConfig:
        default_kwargs = {
            'trust_remote_code': self.config.trust_remote_code
        }
        kwargs = default_kwargs | self.config.hf_model_kwargs | kwargs
        return self.hf_config_class.from_pretrained(self.config.hf_path, **kwargs)

    def load_hf_model(self, **kwargs) -> "PreTrainedModel":
        default_kwargs = {
            'low_cpu_mem_usage': True,
            'trust_remote_code': self.config.trust_remote_code
        }
        kwargs = default_kwargs | self.config.hf_model_kwargs | kwargs
        torch_dtype = kwargs.get('torch_dtype', 'auto')
        kwargs['torch_dtype'] = getattr(torch, torch_dtype, torch_dtype)
        return self.hf_model_class.from_pretrained(self.config.hf_path, **kwargs)

    def load_hf_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        default_kwargs = {
            'trust_remote_code': self.config.trust_remote_code
        }
        kwargs = default_kwargs | kwargs
        return AutoTokenizer.from_pretrained(self.config.hf_path, **kwargs)
    
    def construct_hf_model(self, **kwargs) -> PreTrainedModel:
        default_kwargs = {}
        if issubclass(self.hf_model_class, _BaseAutoModelClass):
            default_kwargs['trust_remote_code'] = self.config.trust_remote_code
            
            torch_dtype = self.config.hf_model_kwargs.get('torch_dtype', None)
            if isinstance(torch_dtype, str):
                default_kwargs['torch_dtype'] = getattr(torch, torch_dtype)
            
            if 'attn_implementation' in self.config.hf_model_kwargs:
                default_kwargs['attn_implementation'] = self.config.hf_model_kwargs['attn_implementation']
            
            kwargs = default_kwargs | kwargs
            return self.hf_model_class.from_config(self.hf_config, **kwargs)
        return self.hf_model_class(self.hf_config)

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return hf_state_dict

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return state_dict

    def get_hf_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        context = shutdown_ds_init_context() if isinstance(self.strategy, DeepSpeedStrategy) else nullcontext()

        with context:
            model = self.load_hf_model()
            state_dict = model.state_dict()
            state_dict = self.convert_state_dict_from_hf(state_dict)

        return state_dict

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        if self.config.hf_path is not None:
            return self.get_hf_pre_trained_weights()
        return super().get_pre_trained_weights()


@contextmanager
def shutdown_ds_init_context():
    import deepspeed # type: ignore
    deepspeed.zero.partition_parameters.shutdown_init_context()
    yield
    deepspeed.zero.partition_parameters.restore_init_context()
