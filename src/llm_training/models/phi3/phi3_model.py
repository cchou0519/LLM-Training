import inspect
import math
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.ops.activations import swiglu
from torch import nn
from torchmetrics.text import Perplexity
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM

from llm_training.metrics import ConsumedSamples, ConsumedTokens
from llm_training.models.hf_compat_model import HFCompatModel
from llm_training.ops.cross_entropy_op import cross_entropy, shift_labels
from llm_training.ops.rms_norm_op import rms_norm
from llm_training.utils.decorators import copy_method_signature

from .phi3_config import LitPhi3Config

try:
    from peft import get_peft_model # type: ignore
except ImportError:
    ...

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


class LitPhi3(HFCompatModel):
    config: LitPhi3Config
    layers: list["Phi3DecoderLayer"]

    hf_config_class = Phi3Config
    hf_model_class = Phi3ForCausalLM

    def __init__(self, config: LitPhi3Config) -> None:
        super().__init__(config)

        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
        self.consumed_samples = ConsumedSamples()
        self.consumed_tokens = ConsumedTokens()

    def update_config_with_hf_config(self, hf_config: Phi3Config) -> None:
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

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k.removeprefix('model.'): v for k, v in hf_state_dict.items()}

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {('' if 'lm_head' in k else 'model.') + k: v for k, v in state_dict.items()}

    def configure_model(self) -> None:
        config = self.config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Phi3RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def on_after_configure_model(self) -> None:
        super().on_after_configure_model()

        if self.config.peft_config is not None:
            from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING # type: ignore

            peft_config = self.config.peft_config

            assert not peft_config.is_prompt_learning and not peft_config.is_adaption_prompt
            assert peft_config.peft_type in PEFT_TYPE_TO_TUNER_MAPPING

            self.peft_model = get_peft_model(self, peft_config)

    def __setattr__(self, name: str, value: torch.Tensor | nn.Module) -> None:
        if name == 'peft_model':
            super(nn.Module, self).__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def forward_layer(
        self,
        layer: "Phi3DecoderLayer",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.config.enable_gradient_checkpointing
            and self.config.recompute_granularity == 'full'
            and self.training
        ):
            return torch.utils.checkpoint.checkpoint(
                layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                # use_reentrant=False
                use_reentrant=True
            )
        return layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

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
        
        past_key_values_length = 0
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attention_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
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
                self.config._attention_implementation == 'sdpa'
                and attention_mask.dtype in [torch.float, torch.bfloat16]
            ):
                attention_mask = attention_mask / 10

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = self.forward_layer(
                layer,
                hidden_states,
                attention_mask,
                position_ids
            )
        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits
    
    @copy_method_signature(forward)
    def __call__(): ...

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            logits.flatten(end_dim=1),
            labels.flatten(end_dim=1),
            implementation='flash_attn'
            # implementation='torch'
        )

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        labels = shift_labels(batch['labels'])

        if self.config.neftune_alpha is not None:
            attention_mask = batch['attention_mask']
            inputs_embeds = self.embed_tokens(batch['input_ids'])
            noise = torch.zeros_like(inputs_embeds).uniform_(-1, 1)
            input_lengths = torch.sum(attention_mask, 1)
            delta = noise * attention_mask.unsqueeze(2)
            dims = input_lengths * inputs_embeds.size(-1)
            magnitude = self.config.neftune_alpha / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1)).detach()
            inputs_embeds += delta
            inputs_embeds.requires_grad_()

            logits = self(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None)
            )

            self.log('NEFTune Alpha', self.config.neftune_alpha)
        else:
            logits = self(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None)
            )

        loss = self.compute_loss(logits, labels)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)

        if self.grad_norm is not None:
            self.log('Gradient Norm', self.grad_norm)

        self.train_perplexity(logits, labels)
        self.log('Perplexity/Train/Step', self.train_perplexity)

        self.consumed_samples.update(labels)
        self.consumed_tokens.update(labels)
        self.logger.log_metrics({
            'Consumed Samples': self.consumed_samples.compute(),
            'Consumed Tokens': self.consumed_tokens.compute()
        })
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['input_ids'].size(0)
        labels = shift_labels(batch['labels'])
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch.get('position_ids', None)
        )

        loss = self.compute_loss(logits, labels)

        self.val_perplexity.update(logits, labels)
        self.log('Loss/Val', loss, batch_size=batch_size, sync_dist=True)
        # self.log('Perplexity/Val', self.val_perplexity)
    
    def on_validation_epoch_end(self) -> None:
        # self.log('Perplexity/Val', self.val_perplexity.compute())
        self.logger.log_metrics({
            'Perplexity/Val': self.val_perplexity.compute()
        })
        self.val_perplexity.reset()


class Phi3RMSNorm(nn.Module):
    def __init__(self, config: LitPhi3Config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps
        self.implementation = config.rms_norm_implementation

        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            x,
            weight=self.weight,
            eps=self.eps,
            implementation=self.implementation
        )


class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3SuScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base ** inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3YarnScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = 0.1 * math.log(scale) + 1.0

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        # self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(swiglu(gate, up_states))
        # return self.down_proj(up_states * F.silu(gate))


@torch.jit.script
def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
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
    def __init__(self, config: LitPhi3Config, layer_idx: int | None = None):
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
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "su":
                self.rotary_emb = Phi3SuScaledRotaryEmbedding(self.head_dim, self.config)
            elif scaling_type == "yarn":
                self.rotary_emb = Phi3YarnScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None
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

        kv_seq_len = key_states.size(-2)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None
    ) -> torch.Tensor:
        # Phi3FlashAttention2 attention does not support output_attentions

        if not _flash_supports_window_size:
            raise ValueError("The current flash attention version does not support sliding window attention.")

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

        kv_seq_len = key_states.size(-2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        kv_seq_len = key_states.size(2)
        
        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

            # logger.warning_once(
            #     f"The input hidden states seems to be silently casted in float32, this might be related to"
            #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            #     f" {target_dtype}."
            # )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        
        causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output
    
    def _get_unpad_data(self, attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = self._get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None
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

        kv_seq_len = key_states.size(-2)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
    def __init__(self, config: LitPhi3Config, layer_idx: int):
        super().__init__()

        self.config = config

        attention_class = Phi3Attention
        if self.config._attention_implementation == 'flash_attention_2':
            attention_class = Phi3FlashAttention2
        elif self.config._attention_implementation == 'sdpa':
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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return hidden_states
