from typing import Protocol

import torch
from torch import nn

from llm_training.models.utils.modeling_outputs import CausalLMOutput


class CausalLMProto(Protocol):
    def get_input_embeddings(self) -> nn.Embedding: ...
    
    def get_output_embeddings(self) -> nn.Linear: ...

    def set_input_embeddings(self, embedding: nn.Embedding) -> None: ...

    def set_output_embeddings(self, linear: nn.Linear) -> None: ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_last_hidden_states: bool = False
    ) -> CausalLMOutput: ...
