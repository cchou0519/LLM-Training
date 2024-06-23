import torch
from .metric import Metric


class ConsumedSamples(Metric):
    higher_is_better: bool = True
    full_state_update: bool = False
    
    n: torch.Tensor

    def __init__(self) -> None:
        super().__init__()

        self.add_state('n', torch.tensor(0), dist_reduce_fx='sum', persistent=True)
    
    def update(self, target: torch.Tensor) -> None:
        self.n += target.size(0)
    
    def compute(self) -> torch.Tensor:
        return self.n
