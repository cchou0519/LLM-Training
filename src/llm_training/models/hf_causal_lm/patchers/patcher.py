from typing import Any, TypeVar

from typing_extensions import Self

T = TypeVar('T')


class Patcher:
    patchers: list[Self] = []

    def __init_subclass__(cls) -> None:
        cls.patchers.append(cls)

    @classmethod
    def match(cls, model: T) -> bool:
        raise NotImplementedError()

    @classmethod
    def patch(cls, model: T, config: dict[str, Any]) -> T:
        raise NotImplementedError()


class AutoPatcher:
    @classmethod
    def patch(cls, model: T, config: dict[str, Any] | None = None) -> T:
        for patcher in Patcher.patchers:
            if patcher.match(model):
                model = patcher.patch(model, config)
                break
        return model
