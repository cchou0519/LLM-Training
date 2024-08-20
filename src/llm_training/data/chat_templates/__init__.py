from pathlib import Path


class _ChatTemplates:
    def _get_path_by_name(self, name: str) -> Path:
        return Path(__file__).parent.joinpath(name).with_suffix('.j2')
    
    def __getitem__(self, name: str) -> str:
        if name not in self:
            raise KeyError(f'Template `{name}` is not found')
        
        with open(self._get_path_by_name(name)) as f:
            return f.read()

    def __contains__(self, name: str) -> bool:
        return self._get_path_by_name(name).exists()


CHAT_TEMPLATES = _ChatTemplates()
