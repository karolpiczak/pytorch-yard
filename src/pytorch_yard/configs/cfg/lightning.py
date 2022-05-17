from dataclasses import dataclass
from typing import Any, Optional

from . import Settings


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = (
        True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    )
    enable_checkpointing: bool = True
    callbacks: Any = None  # Optional[List[Callback]]
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    # num_processes: int = 1
    # gpus: Any = None  # Union[int, str, List[int], NoneType]
    devices: Any = None  # Union[int, str, List[int], NoneType]
    auto_select_gpus: bool = False
    # tpu_cores: Any = None  # Union[int, str, List[int], NoneType]
    log_gpu_memory: Optional[str] = None
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]]
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: int = -1
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    sync_batchnorm: bool = False
    precision: int = 32
    # weights_summary: Optional[str] = "top"
    # weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    # resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    profiler: Any = None  # Union[BaseProfiler, bool, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    # reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    auto_scale_batch_size: Any = False  # Union[str, bool]
    plugins: Any = None  # Union[str, list, NoneType]
    amp_backend: str = "native"
    # amp_level: str = "O2"
    move_metrics_to_cpu: bool = False
    detect_anomaly: bool = False


@dataclass
class LightningConf(TrainerConf):
    deterministic: bool = True
    accelerator: str = "gpu"
    devices: int = 1


@dataclass
class LightningSettings(Settings):
    # Lightning trainer settings
    pl: LightningConf = LightningConf()

    # Additional experiment settings
    validate_before_training: bool = True
    save_checkpoints: bool = True
