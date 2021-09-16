import re
from typing import Any, Optional, Type

from rich import box
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskID,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from torch import Tensor

from avalanche.evaluation.metric_results import MetricValue, TensorImage
from avalanche.evaluation.metric_utils import phase_and_task, stream_type
from avalanche.logging import StrategyLogger
from avalanche.training.strategies.base_strategy import BaseStrategy

UNSUPPORTED_TYPES: tuple[Type[Any]] = (TensorImage, )


class RichLogger(StrategyLogger):
    def __init__(self, ignore_metrics: Optional[list[str]] = None, start_from_epoch_one: bool = True,
                 start_from_experience_one: bool = True):
        super().__init__()
        self.metric_vals: dict[str, Any] = {}
        self.console = Console()

        self.ignore_metrics = ignore_metrics or []
        self.start_from_epoch_one = start_from_epoch_one
        self.start_from_experience_one = start_from_experience_one

        self._pbar: Optional[Progress] = None
        self._pbar_task: Optional[TaskID] = None

        self._reset_table()

    def _reset_table(self):
        self._table = Table(show_header=False, show_edge=True, box=box.ROUNDED)
        self._table.add_column('Metric', style='white', min_width=40)
        self._table.add_column('Value', style='bold magenta')

    def log_metric(self, metric_value: 'MetricValue', callback: str) -> None:
        name = metric_value.name
        x = metric_value.x_plot
        val = metric_value.value
        self.metric_vals[name] = (name, x, val)

    def _val_to_str(self, value: Any):
        if isinstance(value, Tensor):
            return '\n' + str(value)
        elif isinstance(value, float):
            return f'{value:.4f}'
        else:
            return str(value)

    def write_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        for name, _, val in sorted_vals:
            ignored = False
            for regex in self.ignore_metrics:
                if re.match(regex, name):
                    ignored = True
                    break
            if isinstance(val, UNSUPPORTED_TYPES) or ignored:
                continue
            val = self._val_to_str(val)
            self._table.add_row(name, val)

    def flush_metrics(self):
        if self._table.row_count:
            self.console.print(self._table)
        self._reset_table()

    def before_training(self, strategy: BaseStrategy,
                        metric_values: list[MetricValue], **kwargs: Any):
        super().before_training(strategy, metric_values, **kwargs)  # type: ignore
        self.console.print('\n[bold yellow]» :runner: Start of training phase...')

    def before_training_exp(self, strategy: BaseStrategy,
                            metric_values: list[MetricValue], **kwargs: Any):
        super().before_training_exp(strategy, metric_values, **kwargs)  # type: ignore
        self._on_start(strategy)

    def before_training_epoch(self, strategy: BaseStrategy,
                              metric_values: list[MetricValue], **kwargs: Any):
        super().before_training_epoch(strategy, metric_values, **kwargs)  # type: ignore
        self._progress.update(self._main_task, total=len(strategy.dataloader))  # type: ignore

        epoch, experience = self.get_epoch_experience(strategy)
        total = strategy.train_epochs

        self.console.rule(f'[bold yellow]Epoch {epoch} of {total} [cyan]|[yellow] Experience {experience}')
        self.console.print(f'[bold cyan]» :runner: Training (epoch {epoch})...')

        self._on_start(strategy)

    def after_training_iteration(self, strategy: BaseStrategy,
                                 metric_values: list[MetricValue], **kwargs: Any):
        self._progress.advance(self._main_task)
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)  # type: ignore

    def after_training_epoch(self, strategy: BaseStrategy,
                             metric_values: list[MetricValue], **kwargs: Any):
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)  # type: ignore
        self.write_metrics()
        self.flush_metrics()
        self.metric_vals = {}
        # self.console.print(
        # f'[bold green]» :checkered_flag: Training epoch {strategy.epoch} completed. :checkered_flag:')

    def after_training(self, strategy: BaseStrategy,
                       metric_values: list[MetricValue], **kwargs: Any):
        super().after_training(strategy, metric_values, **kwargs)  # type: ignore
        self.console.print('[bold green]» :checkered_flag: End of training phase. :checkered_flag:')

    def before_eval(self, strategy: BaseStrategy,
                    metric_values: list[MetricValue], **kwargs: Any):
        super().before_eval(strategy, metric_values, **kwargs)  # type: ignore

        epoch, _ = self.get_epoch_experience(strategy)
        self.console.print(f'\n[bold yellow]» :magnifying_glass_tilted_left: Evaluation (epoch {epoch})...')

    def before_eval_exp(self, strategy: BaseStrategy,
                        metric_values: list[MetricValue], **kwargs: Any):
        super().before_eval_exp(strategy, metric_values, **kwargs)  # type: ignore
        self._on_start(strategy)
        self._progress.update(self._main_task, total=len(strategy.dataloader))  # type: ignore

    def after_eval_iteration(self, strategy: BaseStrategy,
                             metric_values: list[MetricValue], **kwargs: Any):
        self._progress.advance(self._main_task)
        self._progress.refresh()
        super().after_eval_iteration(strategy, metric_values, **kwargs)  # type: ignore

    def after_eval_exp(self, strategy: BaseStrategy,
                       metric_values: list[MetricValue], **kwargs: Any):
        self._end_progress()
        super().after_eval_exp(strategy, metric_values, **kwargs)  # type: ignore

        self.write_metrics()
        self.metric_vals = {}

    def after_eval(self, strategy: BaseStrategy,
                   metric_values: list[MetricValue], **kwargs: Any):
        super().after_eval(strategy, metric_values, **kwargs)  # type: ignore
        self.write_metrics()
        self.flush_metrics()
        self.metric_vals = {}
        # self.console.print('[bold yellow]» :checkered_flag: [green]End of eval phase. :checkered_flag:')

    def get_epoch_experience(self, strategy: BaseStrategy):
        if strategy.epoch is None:
            return None, None

        epoch = strategy.epoch + 1 if self.start_from_epoch_one else strategy.epoch
        experience: int = strategy.experience.current_experience  # type: ignore
        if self.start_from_experience_one:
            experience += 1

        return epoch, experience

    def _on_start(self, strategy: BaseStrategy):
        action_name = 'train' if strategy.is_training else 'eval'
        _, experience = self.get_epoch_experience(strategy)

        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)  # type: ignore
        if task_id is None:
            description = f'[cyan] | [bold white]{action_name}[normal] » {stream} » exp {experience}'
        else:
            description = f'[cyan] | [bold white]{action_name}[normal] » {stream} » exp {experience} (task {task_id})'
        self._progress.update(self._main_task, description=description)

    @property
    def _progress(self):
        if self._pbar is None:
            columns = [
                "  [progress.percentage][{task.completed} / {task.total}]",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                SpinnerColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                "[progress.description]{task.description}",
            ]

            self._pbar = Progress(*columns, console=self.console, transient=True)
            self._pbar_task = self._pbar.add_task('', total=0)
            self._pbar.start()

        return self._pbar

    @property
    def _main_task(self):
        assert self._pbar_task is not None
        return self._pbar_task

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.stop()
            self._pbar = None
            self._pbar_task = None
