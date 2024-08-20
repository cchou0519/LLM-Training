from lightning import Trainer as _Trainer


class Trainer(_Trainer):
    @property
    def estimated_stepping_batches(self) -> int | float:
        has_train_dataloader = self.train_dataloader is None
        r = super().estimated_stepping_batches
        if not has_train_dataloader:
            self.fit_loop._combined_loader = None
        return r
