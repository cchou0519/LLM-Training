from functools import wraps

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Placement
from torch.distributed.tensor.parallel import *


class PrepareModuleKeywordInput(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: dict[str, Placement],
        desired_input_layouts: dict[str, Placement],
        use_local_output: bool = False
    ):
        self.input_layouts = input_layouts
        self.desired_input_layouts = desired_input_layouts
        self.use_local_output = use_local_output
        assert len(self.input_layouts.keys() - self.desired_input_layouts.keys()) == 0, \
            "input_layouts and desired_input_layouts should have same keys!"
    
    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        old_forward = module.forward
        @wraps(module.forward)
        def forward(*args, **kwargs):
            for k, v in kwargs.items():
                if k in self.input_layouts and isinstance(v, torch.Tensor):
                    input_layout = self.input_layouts[k]
                    desired_layout = self.desired_input_layouts[k]
                    
                    dt = v
                    if not isinstance(v, DTensor):
                        dt = DTensor.from_local(v, device_mesh, (input_layout,), run_check=False)

                    if input_layout != desired_layout:
                        dt = dt.redistribute(placements=(desired_layout,))

                    kwargs[k] = dt.to_local() if self.use_local_output else dt

            return old_forward(*args, **kwargs)
        module.forward = forward
        return module
