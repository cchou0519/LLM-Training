from typing import Any

from transformers.models.phi.modeling_phi import *

from .common import *
from .patcher import Patcher


class PhiPatcher(Patcher):
    @classmethod
    def match(cls, model: Any) -> bool:
        return isinstance(model, PhiForCausalLM)
    
    @classmethod
    def patch(cls, model: PhiForCausalLM, config: dict[str, Any]) -> PhiForCausalLM:
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.forward = fused_layer_norm_forward.__get__(m)
            # elif isinstance(m, PhiDecoderLayer):
            #     m.forward = clamp_decoder_output.__get__(m)
        return model


def clamp_decoder_output(
    self: PhiDecoderLayer,
    *args,
    **kwargs
):
    outputs = PhiDecoderLayer.forward(self, *args, **kwargs)
    outputs = (clamp_fp16(outputs[0]), *outputs[1:])
    return outputs
