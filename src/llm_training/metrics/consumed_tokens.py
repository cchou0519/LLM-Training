import torch
from .metric import Metric


class ConsumedTokens(Metric):
    higher_is_better: bool = True
    full_state_update: bool = False
    
    n: torch.Tensor

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()

        self.ignore_index = ignore_index
        self.add_state('n', torch.tensor(0), dist_reduce_fx='sum', persistent=True)
    
    def update(self, target: torch.Tensor) -> None:
        self.n += target.ne(self.ignore_index).sum()
    
    def compute(self) -> torch.Tensor:
        return self.n
