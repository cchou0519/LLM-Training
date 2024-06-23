from functools import update_wrapper
import inspect
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')


def copy_method_signature(ref_method: Callable[P, T], passthrough: bool = True) -> Callable[[Callable], Callable[P, T]]:
    def decorator(method: Callable):
        wrapped = method
        if passthrough:
            def wrapped(self, *args, _mro_idx: int = 0, **kwargs):
                f = getattr(super(type(self).mro()[_mro_idx], self), method.__name__)
                if '_mro_idx' in inspect.signature(f, follow_wrapped=False).parameters:
                    kwargs['_mro_idx'] = _mro_idx + 1
                return f(*args, **kwargs)
        wrapped = update_wrapper(wrapped, method)
        wrapped = update_wrapper(wrapped, ref_method, [], [])
        return wrapped
    return decorator
