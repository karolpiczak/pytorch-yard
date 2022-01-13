from typing import Any, Tuple, cast

import torch
import torch.utils.data

from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario


def incremental_task(
    train: torch.utils.data.Dataset[torch.Tensor],
    test: torch.utils.data.Dataset[torch.Tensor],
    train_transform: Any,
    eval_transform: Any,
    n_experiences: int,
    n_classes: int,
) -> Tuple[NCScenario, int]:
    """
    Incremental task setup.

    Task labels are provided, classes from `0` to `n_classes` split over `n_experiences` (without remapping).
    Most commonly used with a multi-head setup, where each head is trained on a different experience.
    """

    benchmark = nc_benchmark(
        train_dataset=cast(Any, train),
        test_dataset=cast(Any, test),
        n_experiences=n_experiences,
        task_labels=True,
        fixed_class_order=range(n_classes),
        class_ids_from_zero_in_each_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    n_output_classes = n_classes // n_experiences

    return benchmark, n_output_classes


def incremental_domain(
    train: torch.utils.data.Dataset[torch.Tensor],
    test: torch.utils.data.Dataset[torch.Tensor],
    train_transform: Any,
    eval_transform: Any,
    n_experiences: int,
    n_classes: int,
) -> Tuple[NCScenario, int]:
    """
    Incremental domain setup.

    No task labels, classes are remapped to `0, ..., effective_n_classes`, where
    `effective_n_classes` is the number of classes per single experience.
    """

    benchmark = nc_benchmark(
        train_dataset=cast(Any, train),
        test_dataset=cast(Any, test),
        n_experiences=n_experiences,
        task_labels=False,
        fixed_class_order=range(n_classes),
        class_ids_from_zero_in_each_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    n_output_classes = n_classes // n_experiences

    return benchmark, n_output_classes


def incremental_class(
    train: torch.utils.data.Dataset[torch.Tensor],
    test: torch.utils.data.Dataset[torch.Tensor],
    train_transform: Any,
    eval_transform: Any,
    n_experiences: int,
    n_classes: int,
) -> Tuple[NCScenario, int]:
    """
    Incremental class setup.

    No task labels, classes from `0` to `n_classes` split over `n_experiences` (without remapping).
    """

    benchmark = nc_benchmark(
        train_dataset=cast(Any, train),
        test_dataset=cast(Any, test),
        n_experiences=n_experiences,
        task_labels=False,
        fixed_class_order=range(n_classes),
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    n_output_classes = n_classes

    return benchmark, n_output_classes
