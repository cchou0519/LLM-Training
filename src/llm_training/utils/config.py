import dataclasses
from typing import Any

from typing_extensions import dataclass_transform


@dataclass_transform()
class ConfigMeta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        new_cls = super().__new__(cls, name, bases, attrs)
        return dataclasses.dataclass(new_cls)


class ConfigBase(metaclass=ConfigMeta):
    def __post_init__(self): ...

    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, name: str):
        return getattr(self, name)
    
    def __setitem__(self, name: str, value: Any):
        setattr(self, name, value)

    def to_dict(self):
        return dataclasses.asdict(self)

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)
