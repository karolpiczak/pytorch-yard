import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, cast

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from ..configs import get_tags
from ..configs.cfg.lightning import LightningSettings
from ..core import Experiment
from ..utils.logging import error, info, info_bold


class LightningExperiment(Experiment):
    def before_entry(self) -> None:
        self.cfg: LightningSettings

        self.wandb_logger: WandbLogger
        self.datamodule: Optional[pl.LightningDataModule] = None
        self.system: Optional[pl.LightningModule] = None
        self.callbacks: list[Any] = []
        self.trainer: Optional[pl.Trainer] = None

        self.wandb_logger = WandbLogger(
            project=os.getenv('WANDB_PROJECT'),
            entity=os.getenv('WANDB_ENTITY'),
            name=os.getenv('RUN_NAME'),
            save_dir=str(Path(os.getenv('RUN_DIR', '.'))),
        )

        # Init logger from source dir (code base) before switching to run dir (results)
        self.wandb_logger.experiment  # type: ignore
        self.run: Run = self.wandb_logger.experiment  # type: ignore

    def main(self) -> None:
        tags = get_tags(cast(DictConfig, self.root_cfg))
        self.run.tags = tags
        self.run.notes = str(self.root_cfg.notes)
        self.wandb_logger.log_hyperparams(OmegaConf.to_container(self.root_cfg.cfg, resolve=True))  # type: ignore

        self.setup_datamodule()
        assert self.datamodule is not None, 'A LightningDataModule should be assigned to self.datamodule inside setup_datamodule().'

        self.setup_system()
        assert self.system is not None, 'A main LightningModule should be assigned to self.system inside setup_system().'
        info_bold(f'System architecture:')
        info(str(self.system))

        self.setup_callbacks()

        self.setup_trainer()
        assert self.trainer is not None

        self.fit()

    @abstractmethod
    def setup_datamodule(self) -> None:
        pass
        # TODO: Add datamodule instantiation?
        # datamodule: LightningDataModule = hydra.utils.instantiate(
        #     cfg.experiment.datamodule,
        #     batch_size=cfg.experiment.batch_size,
        #     seed=cfg.experiment.seed,
        #     shuffle=cfg.experiment.shuffle,
        #     num_workers=cfg.experiment.num_workers
        # )

    @abstractmethod
    def setup_system(self) -> None:
        pass

    def setup_callbacks(self) -> None:
        # TODO: Add validation scheduler
        # https://github.com/BioNN-InfoTech/continual-learning-3d-points/blob/main/src/utils/callbacks.py

        if self.cfg.save_checkpoints:
            # TODO: Add custom checkpointer
            # https://github.com/BioNN-InfoTech/continual-learning-3d-points/blob/main/src/utils/callbacks.py
            checkpointer = ModelCheckpoint(
                every_n_epochs=1,  # checkpointing interval in epochs, but still will save only on validation epoch
                dirpath='checkpoints',
                filename='{epoch}',
            )

            self.callbacks.append(checkpointer)

    def setup_trainer(self) -> None:
        num_sanity_val_steps = -1 if self.cfg.validate_before_training else 0

        # TODO: Add resume support
        # resume_path = get_resume_checkpoint(cfg, wandb_logger)
        # if resume_path is not None:
        #     log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

        info_bold(f'Overriding cfg.pl settings with derived values:')
        # info(f' >>> resume_from_checkpoint = {resume_path}')
        info(f' >>> num_sanity_val_steps = {num_sanity_val_steps}')
        info(f'')

        self.trainer = hydra.utils.instantiate(  # type: ignore
            self.cfg.pl,
            logger=self.wandb_logger,
            callbacks=self.callbacks,
            checkpoint_callback=self.cfg.save_checkpoints,
            #     resume_from_checkpoint=resume_path,
            num_sanity_val_steps=num_sanity_val_steps,
        )

    def fit(self) -> None:
        assert self.system
        self.trainer.fit(self.system, datamodule=self.datamodule)  # type: ignore

    def finish(self) -> None:
        if self.trainer.interrupted:  # type: ignore
            error(f'Training interrupted.')
            self.run.finish(exit_code=255)
        else:
            self.run.finish()
