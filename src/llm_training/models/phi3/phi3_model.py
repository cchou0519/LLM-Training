import math

import torch
import torch.distributed
import torch.utils
import torch.utils.checkpoint
from torch import nn
from transformers import Phi3Config as HFPhi3Config
from transformers import Phi3ForCausalLM

from llm_training.models.hf_compat_model import HFCompatModel
from llm_training.ops.attention_op import (flash_attention_forward,
                                           prepare_4d_causal_attention_mask)
from llm_training.ops.liger import *
from llm_training.ops.rope_utils import ROPE_INIT_FUNCTIONS, RoPEConfig
from llm_training.utils.decorators import copy_method_signature

from .phi3_config import Phi3Config


class Phi3(HFCompatModel):
    config: Phi3Config
    layers: list["Phi3DecoderLayer"]

    config_class = Phi3Config
    hf_config_class = HFPhi3Config
    hf_model_class = Phi3ForCausalLM

    no_split_modules = ['Phi3DecoderLayer']

    def __init__(self, config: Phi3Config) -> None:
        super().__init__(config)

        config = self.config
        self.rotary_emb = Phi3RotaryEmbedding(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Phi3RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def merge_hf_config(self, hf_config: HFPhi3Config) -> None:
        assert hf_config.hidden_act == 'silu'
        assert not hf_config.tie_word_embeddings

        self.config.vocab_size = hf_config.vocab_size
        self.config.hidden_size = hf_config.hidden_size
        self.config.intermediate_size = hf_config.intermediate_size
        self.config.num_hidden_layers = hf_config.num_hidden_layers
        self.config.num_attention_heads = hf_config.num_attention_heads
        self.config.num_key_value_heads = hf_config.num_key_value_heads
        self.config.resid_pdrop = hf_config.resid_pdrop
        self.config.embd_pdrop = hf_config.embd_pdrop
        self.config.attention_dropout = hf_config.attention_dropout
        self.config.max_position_embeddings = hf_config.max_position_embeddings
        self.config.original_max_position_embeddings = hf_config.original_max_position_embeddings
        self.config.initializer_range = hf_config.initializer_range
        self.config.rms_norm_eps = hf_config.rms_norm_eps
        self.config.rope_theta = hf_config.rope_theta
        self.config.rope_scaling = hf_config.rope_scaling
        self.config.pad_token_id = hf_config.pad_token_id
        self.config.bos_token_id = hf_config.bos_token_id
        self.config.eos_token_id = hf_config.eos_token_id
        self.config.sliding_window = hf_config.sliding_window

    def _init_weights_impl(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k.removeprefix('model.'): v for k, v in hf_state_dict.items()}

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {('' if 'lm_head' in k else 'model.') + k: v for k, v in state_dict.items()}

    def forward_decoder_layer(
        self,
        layer: "Phi3DecoderLayer",
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.config.enable_gradient_checkpointing
            and self.config.recompute_granularity == 'full'
            and self.training
        ):
            return torch.utils.checkpoint.checkpoint(
                layer.__call__,
                hidden_states,
                position_embeddings,
                attention_mask,
                use_reentrant=False # avoid multiple gradient reduction issue
                # use_reentrant=True
            )
        return layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def get_outut_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                0,
                sliding_window=self.config.sliding_window,
            )

            original_dtype = attention_mask.dtype
            target_dtype = self.config.attention_compute_dtype
            if target_dtype is not None and target_dtype != original_dtype:
                attention_mask = attention_mask.float()
                attention_mask = attention_mask.masked_fill(
                    attention_mask == torch.finfo(original_dtype).min,
                    torch.finfo(target_dtype).min
                )
                attention_mask = attention_mask.to(target_dtype)

            # SDPA with FP32 or BF16 produces NaNs, this fixes the issue.
            if (
                self.config._attn_implementation == 'sdpa'
                and attention_mask.dtype in [torch.float, torch.bfloat16]
            ):
                attention_mask = attention_mask / 10
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = self.forward_decoder_layer(
                layer,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits
    
    @copy_method_signature(forward)
    def __call__(): ...


class Phi3RMSNorm(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps

        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, weight=self.weight, eps=self.eps)


class Phi3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()

        self.config = config
        
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get('rope_type', config.rope_scaling.get('type'))
            scaling_config = config.rope_scaling.copy()
            scaling_config.setdefault('factor', config.max_position_embeddings / config.original_max_position_embeddings)
        else:
            self.rope_type = 'default'
            scaling_config = None
        
        self.rope_config = RoPEConfig(
            type=self.rope_type,
            base=config.rope_theta,
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.original_max_position_embeddings,
            scaling_config=scaling_config
        )

        self.max_seq_len_cached = 0
        self.original_max_seq_len = config.original_max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.attention_scaling = None
        self.register_buffer('inv_freq', None, persistent=False)

        self._set_cos_sin_cache(
            self.original_max_seq_len,
            device=torch.device('cpu'),
            dtype=torch.float
        )

        self.original_inv_freq = self.inv_freq

    def _set_inv_freq_cache(self, seq_len: int, device: torch.device) -> None:
        if self.rope_type == 'dynamic':
            if seq_len > self.max_seq_len_cached:  # growth
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.rope_config,
                    device=device,
                    seq_len=seq_len
                )
                self.register_buffer('inv_freq', inv_freq, persistent=False)  # TODO joao: may break with compilation
                self.max_seq_len_cached = seq_len
            
            if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
                self.register_buffer('inv_freq', self.original_inv_freq, persistent=False)
                self.max_seq_len_cached = self.original_max_seq_len
        elif self.rope_type == 'longrope':
            if seq_len > self.max_seq_len_cached:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.rope_config,
                    device=device,
                    seq_len=seq_len
                )
                self.register_buffer('inv_freq', inv_freq, persistent=False)
            elif seq_len <= self.original_max_seq_len:
                self.register_buffer('inv_freq', self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        else:
            if self.inv_freq is None or self.attention_scaling is None:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.rope_config,
                    device=device,
                    seq_len=seq_len
                )
                self.register_buffer('inv_freq', inv_freq, persistent=False)

            self.max_seq_len_cached = seq_len
        
        if self.inv_freq.device != device:
            self.inv_freq.data = self.inv_freq.data.to(device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self.max_seq_len_cached and seq_len >= self.original_max_seq_len:
            return

        seq_len = math.ceil(seq_len / 4096) * 4096

        self._set_inv_freq_cache(seq_len, device)
        
        device_type = device.type if isinstance(device.type, str) and device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        cos = cos.to(dtype=dtype, device=device)
        sin = sin.to(dtype=dtype, device=device)

        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max() + 1

        self._set_cos_sin_cache(
            seq_len.item(),
            device=x.device,
            dtype=x.dtype
        )
        
        return (
            self.cos[position_ids].to(x),
            self.sin[position_ids].to(x)
        )


class Phi3MLP(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        # self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return swiglu(
            hidden_states,
            w1w2=self.gate_up_proj.weight,
            w3=self.down_proj.weight
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    return hidden_states.repeat_interleave(n_rep, dim=1)


@torch.jit.script
def fused_core_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_dropout: float,
    head_dim: int,
    num_heads: int,
    hidden_size: int,
    training: bool
) -> torch.Tensor:
    bsz = query_states.size(0)
    q_len = query_states.size(2)
    kv_seq_len = key_states.size(2)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=training)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    return attn_output


class Phi3Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)

    def _core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return fused_core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attention_dropout=self.attention_dropout,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            training=self.training
        )

    def core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.config.attention_compute_dtype is not None:
            original_dtype = query_states.dtype
            target_dtype = self.config.attention_compute_dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if (
            self.config.enable_gradient_checkpointing
            and self.config.recompute_granularity == 'selective'
            and self.training
        ):
            attn_output = torch.utils.checkpoint.checkpoint(
                self._core_attention_forward,
                query_states,
                key_states,
                value_states,
                attention_mask,
                use_reentrant=False
            )
        else:
            attn_output = self._core_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask
            )

        if self.config.attention_compute_dtype is not None:
            attn_output = attn_output.to(original_dtype)

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos:query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim:]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask
        )

        attn_output = self.o_proj(attn_output)

        return attn_output


class Phi3FlashAttention2(Phi3Attention):
    """
    Phi-3 flash attention module. This module inherits from `Phi3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Phi3FlashAttention2 attention does not support output_attentions
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask
        )
        
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz = query_states.size(0)
        q_len = query_states.size(2)

        attn_dropout = self.attention_dropout if self.training else 0.0

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.qkv_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            is_causal=self.is_causal,
            dropout=attn_dropout,
            sliding_window=self.config.sliding_window
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output


class Phi3SdpaAttention(Phi3Attention):
    """
    Phi3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Phi3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Phi3Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.core_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask
        )
        
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _core_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz = query_states.size(0)
        q_len = query_states.size(2)
        kv_seq_len = key_states.size(2)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        return attn_output


class Phi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()

        self.config = config

        attention_class = Phi3Attention
        if self.config._attn_implementation == 'flash_attention_2':
            attention_class = Phi3FlashAttention2
        elif self.config._attn_implementation == 'sdpa':
            attention_class = Phi3SdpaAttention

        self.self_attn = attention_class(config, layer_idx=layer_idx)

        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return hidden_states
