from typing import Any

import torch
from transformers.models.phi3.modeling_phi3 import *

from llm_training.ops import rms_norm, swiglu

from .patcher import Patcher

Phi3ForCausalLM._supports_sdpa = True
Phi3Model._supports_sdpa = True


class Phi3Patcher(Patcher):
    @classmethod
    def match(cls, model: Any) -> bool:
        return isinstance(model, Phi3ForCausalLM)
    
    @classmethod
    def patch(cls, model: Phi3ForCausalLM, config: dict[str, Any]) -> Phi3ForCausalLM:
        for m in model.modules():
            if isinstance(m, Phi3RMSNorm):
                m.forward = fused_rms_norm_forward.__get__(m)
            elif isinstance(m, Phi3MLP):
                m.forward = fused_swiglu_forward.__get__(m)
            # elif isinstance(m, Phi3DecoderLayer):
            #     m.forward = clamp_decoder_output.__get__(m)
        return model  


def fused_rms_norm_forward(self: Phi3RMSNorm, x: torch.Tensor) -> torch.Tensor:
    return rms_norm(
        x,
        self.weight,
        eps=self.variance_epsilon,
        implementation='flash_attn'
    )


def fused_swiglu_forward(self: Phi3MLP, x: torch.Tensor) -> torch.Tensor:
    return swiglu(
        x,
        w1w2=self.gate_up_proj.weight,
        w3=self.down_proj.weight,
        implementation='flash_attn'
    )


def clamp_decoder_output(
    self: Phi3DecoderLayer,
    *args,
    **kwargs
):
    outputs = Phi3DecoderLayer.forward(self, *args, **kwargs)
    hidden_states = outputs[0]
    if hidden_states.dtype == torch.half:
        max_dtype = torch.finfo(torch.half).max
        clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    outputs = (hidden_states, *outputs[1:])
    return outputs
