import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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

def get_chat_template(chat_template: str) -> str:
    if Path(chat_template).exists():
        logger.info(f'Found template file at `{chat_template}`.')
        with open(chat_template) as f:
            chat_template = f.read()
    elif chat_template in CHAT_TEMPLATES:
        logger.info(f'Using pre-defined chat template `{chat_template}`.')
        chat_template = CHAT_TEMPLATES[chat_template]
    else:
        logger.warn(
            '`chat_template` is being used directly as a chat template.\n'
            'If this is not the behavior you expected, please change the value to a name of pre-defined chat template or a file path.'
        )
