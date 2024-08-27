from typing import Protocol

import torch
from torch import nn

class CausalLMProto(Protocol):
    def get_input_embeddings(self) -> nn.Embedding: ...
    
    def get_outut_embeddings(self) -> nn.Linear: ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None
    ) -> torch.Tensor: ...

