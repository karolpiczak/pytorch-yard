from typing import Any, Optional, cast

import numpy as np
from matplotlib.figure import Figure
from numpy import array
from PIL.Image import Image
from torch import Tensor

import wandb
import wandb.viz
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience
from avalanche.evaluation.metric_results import (AlternativeValues,
                                                 MetricValue, TensorImage)
from avalanche.logging.strategy_logger import StrategyLogger
from avalanche.training.strategies.base_strategy import BaseStrategy


class WandbEpochLogger(StrategyLogger):
    def __init__(self,
                 project: Optional[str] = None,
                 entity: Optional[str] = None,
                 name: Optional[str] = None,
                 dir: Optional[str] = None,
                 config: Optional[dict[str, Any]] = None,
                 tags: Optional[list[str]] = None,
                 notes: Optional[str] = None,
                 group: Optional[str] = None,
                 start_from_epoch_one: bool = True,
                 start_from_experience_one: bool = True
                 ):
        super().__init__()

        self.start_from_epoch_one = start_from_epoch_one
        self.start_from_experience_one = start_from_experience_one

        self.global_epoch = 1 if self.start_from_epoch_one else 0
        self.experience_epoch = 1 if self.start_from_epoch_one else 0

        self.current_experience = 1 if self.start_from_experience_one else 0

        self.wandb = wandb
        self.wandb.init(project=project, entity=entity, name=name, dir=dir,  # type: ignore
                        config=config, tags=tags, notes=notes, group=group)

    def log_metric(self, metric_value: MetricValue, callback: str):
        name = metric_value.name
        value = metric_value.value

        supported_types = (Image, Tensor, TensorImage, Figure,
                           float, int, self.wandb.Histogram, self.wandb.viz.CustomChart)

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(*supported_types)

        if not isinstance(value, supported_types):
            return  # Unsupported type

        if isinstance(value, Image):
            value = self.wandb.Image(value)
        elif isinstance(value, Tensor):
            value = np.histogram(value.view(-1).numpy())
            value = self.wandb.Histogram(np_histogram=value)
        elif isinstance(value, TensorImage):
            value = self.wandb.Image(np.moveaxis(array(value), 0, -1))

        self.wandb.log({  # type: ignore
            name: value,
            'epoch': self.global_epoch - 1 if 'eval' in callback else self.global_epoch,
            'experience_epoch': self.experience_epoch - 1 if 'eval' in callback else self.experience_epoch,
            'current_experience': self.current_experience,
        }, step=self.global_epoch - 1 if 'eval' in callback else self.global_epoch)

    def before_training(self, strategy: BaseStrategy, metric_values: list[MetricValue], **kwargs: Any):
        super().before_training(strategy, metric_values, **kwargs)  # type: ignore

    def after_training_epoch(self, strategy: BaseStrategy, metric_values: list[MetricValue], **kwargs: Any):
        super().after_training_epoch(strategy, metric_values, **kwargs)  # type: ignore

        self.global_epoch += 1
        self.experience_epoch += 1

    def before_training_exp(self, strategy: BaseStrategy, metric_values: list[MetricValue], **kwargs: Any):
        super().before_training_exp(strategy, metric_values, **kwargs)  # type: ignore

        experience = cast(NCExperience, strategy.experience)  # type: ignore

        self.current_experience = experience.current_experience
        if self.start_from_experience_one:
            self.current_experience += 1
        self.experience_epoch = 1 if self.start_from_epoch_one else 0
