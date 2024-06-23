from jsonargparse import class_from_function
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _load_tokenizer(path: str, pad_token: str | None = None, **kwargs) -> PreTrainedTokenizerBase:
    if pad_token is not None:
        kwargs['pad_token'] = pad_token
    return AutoTokenizer.from_pretrained(path, **kwargs)


HFTokenizer = class_from_function(_load_tokenizer, name='HFTokenizer')
