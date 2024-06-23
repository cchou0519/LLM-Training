import os
from typing import Any, Literal, Optional, Union

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.wandb import WandbLogger as _WandbLogger
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


class WandbLogger(_WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        entity: str | None = None,
        tags: list | None = None,
        save_code: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            entity=entity,
            tags=tags,
            save_code=save_code,
            **kwargs
        )

    @property
    def experiment(self) -> Run:
        os.makedirs(self._wandb_init['dir'], exist_ok=True)
        return super().experiment

    @property
    def name(self) -> str:
        return self._name

    @property
    def save_dir(self) -> str:
        return os.path.join(
            self._save_dir,
            self._project,
            self._name
        )
    
    @property
    def log_dir(self) -> str:
        return os.path.join(
            self._save_dir,
            self._project,
            self._name
        )
