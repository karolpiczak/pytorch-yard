from contextlib import contextmanager
from typing import Any, Optional, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loggers.wandb import WandbLogger
from wandb.sdk.wandb_run import Run


class LightningModuleWithWandb(pl.LightningModule):
    """
    Vanilla LightningModule with WandbLogger experiment pinned to self.wandb.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.wandb: Optional[Run] = None

    @property
    def logger(self) -> LightningLoggerBase:
        """Reference to the logger object in the Trainer."""
        return self.trainer.logger if self.trainer else None  # type: ignore

    @property
    def epoch(self) -> int:
        return self.current_epoch + 1 if not self.trainer.sanity_checking else self.current_epoch  # type: ignore

    def log_wandb(self, data: dict[str, Any], step: Optional[int] = None):
        if self.wandb is not None:
            self.wandb.log(data, step=step)  # type: ignore

    def on_fit_start(self) -> None:
        if isinstance(self.logger, LoggerCollection):
            for logger in self.logger:
                if isinstance(logger, WandbLogger):
                    self.wandb = cast(Run, logger.experiment)  # type: ignore
        elif isinstance(self.logger, WandbLogger):
            self.wandb = cast(Run, self.logger.experiment)  # type: ignore

    @contextmanager
    def no_train(self):
        training = self.training
        self.train(False)
        with torch.no_grad():
            yield
        self.train(training)
