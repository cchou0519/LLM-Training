from typing import TYPE_CHECKING, Any, Literal

import torch

from llm_training.models.hf_compat_model import HFCompatModelConfig
from llm_training.ops.rms_norm_op import RMSNormImplementation

if TYPE_CHECKING:
    try:
        from peft import PeftConfig  # type: ignore
    except ImportError: ...
else:
    try:
        from peft import PeftConfig
    except ImportError:
        PeftConfig = dict[str, Any]


try:
    from peft import get_peft_config  # type: ignore
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LitPhi3Config(HFCompatModelConfig):
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

    neftune_alpha: float | None = None

    attention_implementation: Literal['auto', 'eager', 'sdpa', 'flash_attention_2'] = 'auto'
    recompute_granularity: Literal['full', 'selective'] = 'full'
    attention_compute_dtype: torch.dtype | str | None = None
    rms_norm_implementation: RMSNormImplementation = 'torch'

    peft_config: PeftConfig | dict[str, Any] | None = None
    
    @property
    def _attention_implementation(self) -> str:
        if self.attention_implementation == 'auto':
            return 'flash_attention_2' if torch.cuda.get_device_capability()[0] >= 8 else 'sdpa'
        return self.attention_implementation

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.peft_config is None or PEFT_AVAILABLE, (
            "To use the `peft_config`, you must have PEFT installed."
            " Install it by running `pip install peft`."
        )

        if isinstance(self.peft_config, dict):
            self.peft_config = get_peft_config(self.peft_config)

        self._rope_scaling_validation()
    
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["su", "yarn", "longrope"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['su', 'yarn', 'longrope'], got {rope_scaling_type}")
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        if not len(rope_scaling_short_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_long_factor)}"
            )
