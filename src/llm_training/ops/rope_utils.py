import logging
import math
from typing import Any

import torch
from pydantic import BaseModel, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class RoPEConfig(BaseModel):
    type: str
    base: int
    dim: int
    max_position_embeddings: int
    scaling_config: dict[str, Any] | None = None

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        """
        Validate the RoPE config arguments, given a `PretrainedConfig` object
        """
        validation_fn = ROPE_VALIDATION_FUNCTIONS.get(self.type)
        if validation_fn is not None:
            validation_fn(self)
        else:
            logger.warning(
                f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{self.type}'"
            )


def _compute_default_rope_parameters(
    config: RoPEConfig,
    device: torch.device | None = None,
    seq_len: int | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`RoPEConfig`]):
            The RoPE configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (config.base ** (torch.arange(0, config.dim, 2, dtype=torch.int64).float().to(device) / config.dim))
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config: RoPEConfig,
    device: torch.device | None = None,
    seq_len: int | None = None
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config ([`RoPEConfig`]):
            The RoPE configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= config.scaling_config['factor']
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: RoPEConfig,
    device: torch.device | None = None,
    seq_len: int | None = None
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    Args:
        config ([`RoPEConfig`]):
            The RoPE configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
   
    base = config.base
    dim = config.dim
    max_position_embeddings = config.max_position_embeddings
    factor = config.scaling_config['factor']

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: RoPEConfig,
    device: torch.device | None = None,
    seq_len: int | None = None
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)
    Args:
        config ([`RoPEConfig`]):
            The RoPE configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """

    base = config.base
    dim = config.dim
    max_position_embeddings = config.max_position_embeddings
    factor = config.scaling_config['factor']

    # Sets the attention factor as suggested in the paper
    attention_factor = config.scaling_config.get('attention_factor')
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0
    
    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.scaling_config.get('beta_fast') or 32
    beta_slow = config.scaling_config.get('beta_slow') or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range bounds based on rotations"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: RoPEConfig,
    device: torch.device | None = None,
    seq_len: int | None = None
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    Args:
        config ([`RoPEConfig`]):
            The RoPE configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """

    base = config.base
    dim = config.dim
    long_factor = config.scaling_config['long_factor']
    short_factor = config.scaling_config['short_factor']
    factor = config.scaling_config['factor']
    attention_factor = config.scaling_config.get('attention_factor')

    max_position_embeddings = config.max_position_embeddings
    expanded_max_position_embeddings = max_position_embeddings * factor

    seq_len = seq_len or expanded_max_position_embeddings
    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len > max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: RoPEConfig,
    device: torch.device | None = None,
    seq_len: int | None = None
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`RoPEConfig`]):
            The RoPE configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    factor = config.scaling_config['factor']  # `8` in the original implementation
    low_freq_factor = config.scaling_config['low_freq_factor']  # `1` in the original implementation
    high_freq_factor = config.scaling_config['high_freq_factor']  # `4` in the original implementation
    old_context_len = config.scaling_config['original_max_position_embeddings']  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}


def _check_received_keys(rope_type: str, received_keys: set[str], required_keys: set[str], optional_keys: set[str] | None = None):
    """Compare the received keys in `config.scaling_config` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's gracefully handle it
    if 'rope_type' not in received_keys and 'type' in received_keys:
        received_keys -= {'type'}
        received_keys.add('rope_type')

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `scaling_config` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in `scaling_config` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(config: RoPEConfig):
    rope_scaling = config.scaling_config

    if rope_scaling is None:
        return

    rope_type = config.type
    required_keys = {'rope_type'}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys)


def _validate_linear_scaling_rope_parameters(config: RoPEConfig):
    rope_scaling = config.scaling_config
    rope_type = config.type
    required_keys = {'rope_type', 'factor'}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys)

    factor = rope_scaling['factor']
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config: RoPEConfig):
    rope_scaling = config.scaling_config
    rope_type = config.type
    required_keys = {'rope_type', 'factor'}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {'original_max_position_embeddings'}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys)

    factor = rope_scaling['factor']
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(config: RoPEConfig):
    rope_scaling = config.scaling_config
    rope_type = config.type
    required_keys = {'rope_type', 'factor'}
    optional_keys = {'attention_factor', 'beta_fast', 'beta_slow'}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys)

    factor = rope_scaling['factor']
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get('attention_factor')
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_scaling.get('beta_fast')
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get('beta_slow')
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )


def _validate_longrope_parameters(config: RoPEConfig):
    rope_scaling = config.scaling_config
    rope_type = config.type
    required_keys = {'rope_type', 'short_factor', 'long_factor'}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {'attention_factor', 'factor', 'original_max_position_embeddings'}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys)

    dim = config.dim

    short_factor = rope_scaling.get('short_factor')
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        logger.warning(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
    if not len(short_factor) == dim // 2:
        logger.warning(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_scaling.get('long_factor')
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        logger.warning(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
    if not len(long_factor) == dim // 2:
        logger.warning(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    factor = rope_scaling.get('factor')
    if factor is None:
        logger.warning("Missing required keys in `rope_scaling`: 'factor'")
    elif not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get('attention_factor')
    if attention_factor is not None and not isinstance(attention_factor, float) or attention_factor is not None and attention_factor < 0:
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )


def _validate_llama3_parameters(config: RoPEConfig):
    rope_scaling = config.scaling_config
    rope_type = config.type
    required_keys = {'rope_type', 'factor', 'original_max_position_embeddings', 'low_freq_factor', 'high_freq_factor'}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys)

    factor = rope_scaling['factor']
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_scaling['low_freq_factor']
    high_freq_factor = rope_scaling['high_freq_factor']
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        logger.warning(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        logger.warning(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor <= low_freq_factor:
        logger.warning(
            "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = rope_scaling['original_max_position_embeddings']
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        logger.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    
    max_position_embeddings = config.max_position_embeddings
    if original_max_position_embeddings >= max_position_embeddings:
        logger.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={max_position_embeddings}"
        )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}
