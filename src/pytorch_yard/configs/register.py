from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type, cast

import omegaconf
from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.core.config_store import ConfigStore
from omegaconf import SI, DictConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING

from .cfg import Settings


@dataclass
class Hydra(HydraConf):
    run: RunDir = RunDir("${output_dir}")
    sweep: SweepDir = SweepDir(".", "${output_dir}")
    searchpath: List[str] = field(default_factory=lambda: [
        'pkg://pytorch_yard/configs'
    ])


@dataclass
class RootConfig():
    """
    Top-level Hydra config class.
    """
    defaults: List[Any] = field(default_factory=lambda: [
        {'cfg': 'default'},
        '_self_',
        {'override hydra/job_logging': 'rich'},
        {'override hydra/hydra_logging': 'rich'},
    ])

    # Path settings
    data_dir: str = SI("${oc.env:DATA_DIR}")
    output_dir: str = SI("${oc.env:RUN_DIR}")

    # Runtime configuration
    hydra: Hydra = Hydra()

    # Experiment settings --> *.yaml
    cfg: Settings = MISSING

    # wandb metadata
    notes: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None


def register_configs(settings_cls: Optional[Type[Settings]] = None, settings_group: Optional[str] = None):
    """
    Register configuration options in the main ConfigStore.instance().

    The term `config` is used for a StructuredConfig at the root level (normally switchable with `-cn`
    flag in Hydra, here we use only one default config). Fields of the main config use StructuredConfigs
    with class names ending in `Settings`. `Conf` suffix is used for external schemas provided by
    the `hydra-torch` package for PyTorch/PyTorch Lightning integration, e.g. `AdamConf`.
    """
    cs = ConfigStore.instance()

    settings_group = settings_group or 'cfg'

    # Main config
    root = RootConfig()
    root.defaults[0] = {settings_group: 'default'}
    cs.store(name='root', node=DictConfig(root))

    # Config groups with defaults, YAML files validated by Python structured configs
    # e.g.: `python train.py experiment=default`
    cs.store(group=settings_group, name='settings_schema', node=settings_cls if settings_cls is not None else Settings)


def _get_tags(cfg: dict[str, Any]) -> Iterator[str]:
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from _get_tags(cast(Dict[str, Any], value))
        if key == '_tags_':
            if isinstance(value, list):
                for v in cast(List[str], value):
                    yield v
            else:
                if value is not None:
                    value = cast(str, value)
                    yield value


def get_tags(cfg: DictConfig):
    """
    Extract all tags from a nested DictConfig object.
    """
    cfg_dict = cast(Dict[str, Any], omegaconf.OmegaConf.to_container(cfg, resolve=True))
    if 'tags' in cfg_dict:
        cfg_dict['_tags_'] = cfg_dict['tags']
    return list(_get_tags(cfg_dict))
