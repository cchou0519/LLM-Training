from typing import Any, Literal

import torch
from pydantic import ValidationInfo, field_validator

from llm_training.models.hf_compat_model import HFCompatModelConfig


class Phi3Config(HFCompatModelConfig):
    vocab_size: int = 32064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attention_dropout: float = 0.0
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    bos_token_id: int = 1
    eos_token_id: int = 32000
    pad_token_id: int = 32000
    sliding_window: int | None = None

    enable_gradient_checkpointing: bool = False
    recompute_granularity: Literal['full', 'selective'] = 'full'
    attention_compute_dtype: torch.dtype | str | None = None

    @field_validator('rope_scaling')
    @classmethod
    def validate_rope_scaling(cls, rope_scaling: dict[str, Any] | None, info: ValidationInfo) -> dict[str, Any] | None:
        """
        Validate the `rope_scaling` configuration.
        """
        if rope_scaling is None:
            return rope_scaling

        hidden_size = info.data['hidden_size']
        num_attention_heads = info.data['num_attention_heads']

        if not isinstance(rope_scaling, dict) or len(rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {rope_scaling}"
            )
        rope_scaling_type = rope_scaling.get('type', None)
        rope_scaling_short_factor = rope_scaling.get('short_factor', None)
        rope_scaling_long_factor = rope_scaling.get('long_factor', None)
        if rope_scaling_type is None or rope_scaling_type not in ['longrope']:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}")
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        if not len(rope_scaling_short_factor) == hidden_size // num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {hidden_size // num_attention_heads // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == hidden_size // num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {hidden_size // num_attention_heads // 2}, got {len(rope_scaling_long_factor)}"
            )

        return rope_scaling
