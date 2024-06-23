from typing import Any

import torch
import torch.nn.functional as F
from flash_attn.ops.activations import swiglu
from flash_attn.ops.triton.layer_norm import rms_norm_fn
from transformers.models.llama import modeling_llama

from .common import *
from .patcher import Patcher


class LlamaPatcher(Patcher):
    @classmethod
    def match(cls, model: Any) -> bool:
        return isinstance(model, modeling_llama.LlamaForCausalLM)
    
    @classmethod
    def patch(cls, model: modeling_llama.LlamaForCausalLM, config: dict[str, Any]) -> modeling_llama.LlamaForCausalLM:
        for m in model.modules():
            if isinstance(m, modeling_llama.LlamaRMSNorm):
                m.forward = fused_rms_norm_forward.__get__(m)
            elif isinstance(m, modeling_llama.LlamaMLP):
                m.forward = fused_swiglu_forward.__get__(m)
        return model


def fused_rms_norm_forward(self: modeling_llama.LlamaRMSNorm, x: torch.Tensor) -> torch.Tensor:
    return rms_norm_fn(
        x,
        weight=self.weight,
        bias=None,
        eps=self.variance_epsilon
    )


def fused_swiglu_forward(self: modeling_llama.LlamaMLP, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def clamp_decoder_output(
    self: modeling_llama.LlamaDecoderLayer,
    *args,
    **kwargs
):
    outputs = modeling_llama.LlamaDecoderLayer.forward(self, *args, **kwargs)
    outputs = (clamp_fp16(outputs[0]), *outputs[1:])
    return outputs


@torch.jit.script
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2:]
    return torch.cat([-x2, x1], dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.jit.script
def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = attention_mask.flatten().nonzero().flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

modeling_llama.rotate_half = rotate_half
# modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
modeling_llama.repeat_kv = repeat_kv
modeling_llama._get_unpad_data = _get_unpad_data
