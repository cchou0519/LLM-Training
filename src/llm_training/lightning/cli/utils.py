from typing import Literal

from jsonargparse import class_from_function
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _load_tokenizer(
    path: str,
    pad_token: str | None = None,
    padding_side: Literal["left", "right"] | None = None,
    **kwargs
) -> PreTrainedTokenizerBase:
    if pad_token is not None:
        kwargs['pad_token'] = pad_token

    if padding_side is not None:
        kwargs['padding_side'] = padding_side

    return AutoTokenizer.from_pretrained(path, **kwargs)


HFTokenizer = class_from_function(_load_tokenizer, name='HFTokenizer')
