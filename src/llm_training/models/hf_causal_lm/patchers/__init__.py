import importlib.util
from pathlib import Path

from .patcher import AutoPatcher

for file in Path(__file__).parent.glob('*.py'):
    if file.stem in ['__init__', 'patcher', 'common']:
        continue
    spec = importlib.util.spec_from_file_location(f'{__name__}.{file.stem}', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
