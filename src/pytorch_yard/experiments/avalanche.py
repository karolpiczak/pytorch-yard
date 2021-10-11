import os
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_yard.configs.cfg import Settings
from torch import Tensor

from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario

from ..avalanche.wandb import WandbEpochLogger
from ..configs import get_tags
from ..core import Experiment


class AvalancheExperiment(Experiment):
    def before_entry(self) -> None:
        self.cfg: Settings
        """ Experiment config. """

        self.device: torch.device
        """ Device selector (CPU/CUDA). """

        self.train: torch.utils.data.Dataset[Tensor]
        """ Train dataset. """

        self.test: torch.utils.data.Dataset[Tensor]
        """ Test dataset. """

        self.transforms: Any
        """ Transforms applied to train and test data. """

        self.n_classes: int
        """ Number of classes in the dataset. """

        self.scenario: NCScenario
        """ Main scenario object. """

        self.model: nn.Module
        """ Main PyTorch model. """

        self.wandb_logger: WandbEpochLogger
        """ W&B logger object. """

    def main(self) -> None:
        self.wandb_logger = WandbEpochLogger(
            project=os.getenv('WANDB_PROJECT'),
            entity=os.getenv('WANDB_ENTITY'),
            name=os.getenv('RUN_NAME'),
            dir=str(Path(os.getenv('RUN_DIR', '.'))),
            config=OmegaConf.to_container(self.root_cfg, resolve=True),  # type: ignore
            tags=get_tags(cast(DictConfig, self.root_cfg)),
            notes=str(self.root_cfg.notes) if self.root_cfg.notes is not None else None,
            group=str(self.root_cfg.group) if self.root_cfg.group is not None else None,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def finish(self) -> None:
        self.wandb_logger.wandb.finish()  # type: ignore
