import os
from pathlib import Path
from typing import cast

from omegaconf.dictconfig import DictConfig

from ..avalanche.wandb import WandbEpochLogger
from ..configs import get_tags
from ..core import Experiment


class AvalancheExperiment(Experiment):
    def before_entry(self) -> None:
        pass

    def main(self) -> None:
        self.wandb_logger = WandbEpochLogger(
            project=os.getenv('WANDB_PROJECT'),
            entity=os.getenv('WANDB_ENTITY'),
            name=os.getenv('RUN_NAME'),
            dir=str(Path(os.getenv('RUN_DIR', '.'))),
            config=OmegaConf.to_container(self.root_cfg, resolve=True),  # type: ignore
            tags=get_tags(cast(DictConfig, self.root_cfg)),
            notes=str(self.root_cfg.notes),
        )

    def finish(self) -> None:
        self.wandb_logger.wandb.finish()  # type: ignore
