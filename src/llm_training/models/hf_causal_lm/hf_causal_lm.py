from functools import partial

import torch
import torch.distributed
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from transformers import (AutoConfig, AutoModelForCausalLM,
                          modeling_flash_attention_utils)
from transformers.modeling_utils import no_init_weights

from llm_training.models.hf_compat_model import HFCompatModel
from llm_training.models.utils.modeling_outputs import CausalLMOutput
from llm_training.ops.attention_op import _get_unpad_data
from llm_training.utils.decorators import copy_method_signature

from .hf_causal_lm_config import HFCausalLMConfig

# Patch for packed attention masks (FA only)
modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

class HFCausalLM(HFCompatModel):
    config: HFCausalLMConfig

    config_class = HFCausalLMConfig
    hf_model_class = AutoModelForCausalLM
    hf_config_class = AutoConfig

    def __init__(self, config: HFCausalLMConfig) -> None:
        super().__init__(config)

        self.config.hf_config = self.hf_config

        with no_init_weights(not self._init_weights):
            self.hf_model = self.construct_hf_model()

        if self.config.enable_gradient_checkpointing:
            self.hf_model.gradient_checkpointing_enable({'use_reentrant': False})

        self.hf_model.tie_weights()
        
        if self.config.enable_liger_kernel:
            _apply_liger_kernel_to_instance(self.hf_model, rope=False)

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {'hf_model.' + k: v for k, v in hf_state_dict.items()}

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k.removeprefix('hf_model.'): v for k, v in state_dict.items()}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_last_hidden_states: bool = False
    ) -> CausalLMOutput:
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=return_last_hidden_states
        )

        last_hidden_states = None
        if return_last_hidden_states:
            last_hidden_states = outputs.hidden_states[-1]

        return CausalLMOutput(
            logits=outputs.logits,
            last_hidden_states=last_hidden_states
        )

    @copy_method_signature(forward)
    def __call__(): ...

    def get_input_embeddings(self) -> nn.Embedding:
        return self.hf_model.get_input_embeddings()
    
    def get_output_embeddings(self) -> nn.Linear:
        return self.hf_model.get_output_embeddings()

    def configure_fully_sharded_data_parallel(self, dp_mesh, reshard_after_forward, mp_policy, offload_policy, **kwargs):
        if dp_mesh.size() == 1:
            return
        
        fully_shard_ = partial(
            fully_shard,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy
        )
        
        fully_sharded_module_names = []
        no_split_modules = self.hf_model._no_split_modules or []
        for n, m in self.named_modules():
            if m.__class__.__name__ in no_split_modules:
                fully_shard_(m)
                fully_sharded_module_names.append(n)

        # shard the rest modules for gradient clipping as a workaround
        for n, m in self.named_modules():
            for o in fully_sharded_module_names:
                if o.startswith(n) or n.startswith(o):
                    break
            else:
                if any(True for _ in m.parameters()):
                    fully_shard_(m)
