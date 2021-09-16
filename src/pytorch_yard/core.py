import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type, cast

import hydra
import numpy as np
import setproctitle
import torch
from dotenv import load_dotenv
from dotenv.main import find_dotenv
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .configs import RootConfig, Settings, get_tags, register_configs
from .utils.logging import info, info_bold
from .utils.rundir import finish_rundir, setup_rundir

load_dotenv(find_dotenv(usecwd=True))


class Experiment(ABC):
    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        """
        Run an experiment from a provided entry point with minimal boilerplate code.

        Incorporates run directory setup and Hydra support.
        """

        self.settings_cls = settings_cls
        self.settings_group = settings_group
        self.root_cfg: RootConfig

        assert os.getenv('DATA_DIR') is not None, "Missing DATA_DIR environment variable."
        assert os.getenv('RESULTS_DIR') is not None, "Missing RESULTS_DIR environment variable."
        assert os.getenv('WANDB_PROJECT') is not None, "Missing WANDB_PROJECT environment variable."
        os.environ['DATA_DIR'] = str(Path(os.environ['DATA_DIR']).expanduser())
        os.environ['RESULTS_DIR'] = str(Path(os.environ['RESULTS_DIR']).expanduser())

        setup_rundir()
        register_configs(self.settings_cls, self.settings_group)

        # Hydra will change workdir to the run dir before calling `on_init`
        hydra_decorator = hydra.main(config_path=config_path, config_name='root')
        hydra_decorator(self.main)()

        self.on_finish()
        finish_rundir()

    @abstractmethod
    def on_preinit(self) -> None:
        pass

    @abstractmethod
    def on_init(self) -> None:
        pass

    @abstractmethod
    def on_finish(self) -> None:
        pass

    @abstractmethod
    def main(self, root_cfg: RootConfig) -> None:
        """
        Main experiment function. Called after initial setup with `cfg` populated by Hydra.

        Parameters
        ----------
        cfg : Config
            Top-level Hydra config for the experiment.
        """
        self.root_cfg = root_cfg

        RUN_NAME = os.getenv('RUN_NAME')
        info_bold(f'\\[init] Run name --> {RUN_NAME}')
        info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(self.root_cfg, resolve=True)}')
        setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

        self.seed_everything(root_cfg.cfg.seed)
        self.on_init()

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed_all(seed)


class PyTorchExperiment(Experiment):
    def on_preinit(self) -> None:
        pass

    def on_init(self) -> None:
        pass

    def on_finish(self) -> None:
        pass


class AvalancheExperiment(Experiment):
    def on_preinit(self) -> None:
        pass

    def on_init(self) -> None:
        from .avalanche.wandb import WandbEpochLogger

        self.wandb_logger = WandbEpochLogger(
            project=os.getenv('WANDB_PROJECT'),
            entity=os.getenv('WANDB_ENTITY'),
            name=os.getenv('RUN_NAME'),
            dir=str(Path(os.getenv('RUN_DIR', '.'))),
            config=OmegaConf.to_container(self.root_cfg, resolve=True),  # type: ignore
            tags=get_tags(cast(DictConfig, self.root_cfg)),
            notes=str(self.root_cfg.notes),
        )

    def on_finish(self) -> None:
        self.wandb_logger.wandb.finish()  # type: ignore


class LightningExperiment(Experiment):
    def on_preinit(self) -> None:
        pass

    def on_init(self) -> None:
        raise NotImplementedError()

        # pl.seed_everything(cfg.experiment.seed)

     # wandb_logger = WandbLogger(
        #     project=os.getenv('WANDB_PROJECT'),
        #     entity=os.getenv('WANDB_ENTITY'),
        #     name=os.getenv('RUN_NAME'),
        #     save_dir=os.getenv('RUN_DIR'),
        # )

        # # Init logger from source dir (code base) before switching to run dir (results)
        # wandb_logger.experiment  # type: ignore
        # run.tags = tags
        # run.notes = str(cfg.notes)
        # wandb_logger.log_hyperparams()

        #   run: Run = wandb_logger.experiment  # type: ignore

        # # Prepare data using datamodules
        # # https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#using-a-datamodule
        # datamodule: LightningDataModule = instantiate(
        #     cfg.experiment.datamodule,
        #     batch_size=cfg.experiment.batch_size,
        #     seed=cfg.experiment.seed,
        #     shuffle=cfg.experiment.shuffle,
        #     num_workers=cfg.experiment.num_workers
        # )

        # # Create main system (system = models + training regime)
        # system = ImageClassifier(cfg)
        # log.info(f'[bold yellow]\\[init] System architecture:')
        # log.info(system)

        # Setup logging & checkpointing
        # tags = get_tags(cast(DictConfig, cfg))
        # run.tags = tags
        # run.notes = str(cfg.notes)
        # wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
        # log.info(f'[bold yell

        # resume_path = get_resume_checkpoint(cfg, wandb_logger)
        # if resume_path is not None:
        #     log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

        # callbacks: list[Any] = []

        # checkpointer = CustomCheckpointer(
        #     period=1,  # checkpointing interval in epochs, but still will save only on validation epoch
        #     dirpath='checkpoints',
        #     filename='{epoch}',
        # )
        # if cfg.experiment.save_checkpoints:
        #     callbacks.append(checkpointer)

        # log.info(f'[bold white]Overriding cfg.pl settings with derived values:')
        # log.info(f' >>> resume_from_checkpoint = {resume_path}')
        # log.info(f' >>> num_sanity_val_steps = {-1 if cfg.experiment.validate_before_training else 0}')
        # log.info(f'')

        # trainer: pl.Trainer = instantiate(
        #     cfg.pl,
        #     logger=wandb_logger,
        #     callbacks=callbacks,
        #     checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        #     resume_from_checkpoint=resume_path,
        #     num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
        # )

        # trainer.fit(system, datamodule=datamodule)  # type: ignore
        # # Alternative way to call:
        # # trainer.fit(system, train_dataloader=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

        # if trainer.interrupted:  # type: ignore
        #     log.info(f'[bold red]>>> Training interrupted.')
        #     run.finish(exit_code=255)

    def on_finish(self) -> None:
        pass
