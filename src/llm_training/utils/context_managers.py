from contextlib import AbstractContextManager, ExitStack
from typing import ContextManager


class ContextManagers(AbstractContextManager):
    def __init__(self, context_managers: list[ContextManager]) -> None:
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for cm in self.context_managers:
            self.stack.enter_context(cm)
        return self
    
    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.stack.close()
