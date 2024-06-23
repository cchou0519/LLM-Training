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
    # hidden_act: str= "silu"
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    # use_cache: bool = True
    # tie_word_embeddings: bool = False
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
