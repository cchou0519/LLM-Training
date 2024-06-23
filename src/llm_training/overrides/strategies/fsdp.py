from typing import Literal

from lightning.pytorch.strategies.fsdp import FSDPStrategy as _FSDPStrategy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp.api import ShardingStrategy


class FSDPStrategy(_FSDPStrategy):
    def __init__(
        self,
        *args,
        state_dict_type: Literal["full", "sharded"] = 'sharded',
        use_orig_params: bool = False,
        tp_size: int = 1,
        **kwargs,
    ) -> None:
        kwargs.setdefault('device_mesh', ...)
        kwargs['use_orig_params'] = use_orig_params
        kwargs['state_dict_type'] = state_dict_type
        
        super().__init__(*args, **kwargs)

        if kwargs['device_mesh'] is ...:
            del kwargs['device_mesh']

        self._tp_size = tp_size

        assert self._tp_size == 1 or not self.is_hsdp, "HSDP + TP is not supported currently."
    
    @property
    def is_hsdp(self) -> bool:
        return self.sharding_strategy in [ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2]

    @property
    def distributed_sampler_kwargs(self) -> dict:
        return {
            'num_replicas': self.dp_size,
            'rank': self.dp_rank
        }

    @property
    def dp_mesh(self) -> DeviceMesh | None:
        if self.is_hsdp:
            return self.device_mesh
        return self.device_mesh['dp']

    @property
    def dp_size(self) -> int:
        if self.is_hsdp:
            return self.world_size
        return self.dp_mesh.size()

    @property
    def dp_rank(self) -> int:
        if self.is_hsdp:
            return self.global_rank
        return self.dp_mesh.get_local_rank()

    @property
    def tp_mesh(self) -> DeviceMesh | None:
        if self._tp_size == 1:
            return None
        return self.device_mesh['tp']

    @property
    def tp_size(self) -> int:
        if self.tp_mesh is None:
            return 1
        return self.tp_mesh.size()

    @property
    def tp_rank(self) -> int:
        if self.tp_mesh is None:
            return 0
        return self.tp_mesh.get_local_rank()
    
    def get_rank(self, rank: int | None = None, dim: int | str = 0) -> int:
        rank = self.global_rank if rank is None else rank
        dim = self.device_mesh.mesh_dim_names.index(dim) if isinstance(dim, str) else dim
        return (self.device_mesh.mesh == rank).argwhere().flatten()[dim].item()

    def setup_environment(self) -> None:
        super().setup_environment()
        
        if self.is_hsdp:
            self.device_mesh = init_device_mesh(
                self.root_device.type,
                (self.num_nodes, self.num_processes),
                mesh_dim_names=('replicate', 'shard')
            )
            self.kwargs['device_mesh'] = self.dp_mesh
        else:
            assert self._tp_size < self.world_size
            assert self.world_size % self._tp_size == 0

            dp_size = self.world_size // self._tp_size
            self.device_mesh = init_device_mesh(
                self.root_device.type,
                (dp_size, self._tp_size),
                mesh_dim_names=('dp', 'tp')
            )
            self.kwargs['device_mesh'] = self.dp_mesh
