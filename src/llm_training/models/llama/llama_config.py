
from typing import Any, Literal

import torch

from llm_training.models.hf_compat_model import HFCompatModelConfig


class LlamaConfig(HFCompatModelConfig):
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    # hidden_act: str = 'silu'
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = None
    # tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0

    mlp_bias: bool = False
    rope_scaling: dict[str, Any] | None = None

    pad_token_id: int | None = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    enable_gradient_checkpointing: bool = False
    recompute_granularity: Literal['full', 'selective'] = 'full'
