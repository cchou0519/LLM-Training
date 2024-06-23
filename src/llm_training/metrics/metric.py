import torch
from torchmetrics import Metric as _Metric


class Metric(_Metric):
    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str]
    ) -> None:
        for key in self._defaults:
            name = prefix + key
            tensor = getattr(self, key, None)
            if name in state_dict and isinstance(tensor, torch.Tensor):
                tensor.copy_(state_dict.pop(name))
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
