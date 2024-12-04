import torch
from pydantic import BaseModel, ConfigDict


class ModelOutput(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )


class CausalLMOutput(ModelOutput):
    logits: torch.Tensor
    last_hidden_states: torch.Tensor | None = None
