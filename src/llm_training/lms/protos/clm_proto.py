from typing import Protocol

import torch


class CausalLMProto(Protocol):
    def get_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None
    ) -> torch.Tensor: ...

