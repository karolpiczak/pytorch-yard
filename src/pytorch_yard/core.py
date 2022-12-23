import os
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type

import hydra
import lovely_tensors as lt
import numpy.random
import setproctitle  # type: ignore
import torch
from dotenv import load_dotenv  # type: ignore
from dotenv.main import find_dotenv
from git.repo import Repo
from omegaconf import OmegaConf
from rich import print

from .configs import RootConfig, Settings, register_configs
from .utils.logging import info, info_bold
from .utils.rundir import finish_rundir, setup_rundir

load_dotenv(find_dotenv(usecwd=True))
lt.monkey_patch()  # type: ignore


class Experiment(ABC):
    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        """
        Run an experiment from a provided entry point with minimal boilerplate code.

        Incorporates run directory setup and Hydra support.
        """

        self.settings_cls = settings_cls
        self.settings_group = settings_group
        self.root_cfg: RootConfig

        assert os.getenv("DATA_DIR") is not None, "Missing DATA_DIR environment variable."
        assert os.getenv("RESULTS_DIR") is not None, "Missing RESULTS_DIR environment variable."
        assert os.getenv("WANDB_PROJECT") is not None, "Missing WANDB_PROJECT environment variable."
        os.environ["DATA_DIR"] = str(Path(os.environ["DATA_DIR"]).expanduser())
        os.environ["RESULTS_DIR"] = str(Path(os.environ["RESULTS_DIR"]).expanduser())

        setup_rundir()
        self.before_entry()
        self.save_state()

        # Hydra will change workdir to the run dir before calling `self.main`
        register_configs(self.settings_cls, self.settings_group)
        hydra_decorator = hydra.main(config_path=config_path, config_name="root", version_base="1.1")
        hydra_decorator(self.entry)()

        self.finish()
        finish_rundir()

    @abstractmethod
    def before_entry(self) -> None:
        pass

    def save_state(self) -> None:
        try:
            src_repo = Repo(".")
            dst_repo = Repo.clone_from(url=".", to_path=str(Path(f"{os.environ['RUN_DIR']}/code")))  # type: ignore
            dst_repo.git.checkout("-b", os.getenv("RUN_NAME"))

            # Git state
            git_status = f"{str(src_repo.head.commit.hexsha)}\n{str(src_repo.head.commit.message)}"
            Path(f"{os.environ['RUN_DIR']}/code.git").write_text(git_status, encoding="utf-8")

            # Pending changes
            pending_diff: str = src_repo.git.diff("HEAD~0")

            if pending_diff:
                Path(f"{os.environ['RUN_DIR']}/code.diff").write_text(pending_diff + "\n", encoding="utf-8")

                dst_repo.git.apply(["-3", f"{os.environ['RUN_DIR']}/code.diff"])
                dst_repo.index.commit("Save uncommitted code changes")
        except Exception:
            print(":exclamation:[bold yellow]Code state saving failed. Maybe not running from a git repository?")

        # Cmd line
        Path(f"{os.environ['RUN_DIR']}/code.cmd").write_text(" ".join(sys.argv) + "\n")

    @abstractmethod
    def entry(self, root_cfg: RootConfig) -> None:
        """
        Experiment entrypoint. Called after initial setup with `cfg` populated by Hydra.

        Parameters
        ----------
        cfg : Config
            Top-level Hydra config for the experiment.
        """
        self.root_cfg = root_cfg
        self.cfg = root_cfg.cfg

        RUN_NAME = os.getenv("RUN_NAME")
        info_bold(f"\\[init] Run name --> {RUN_NAME}")

        cfg = OmegaConf.to_yaml(self.root_cfg, resolve=True)
        info(f"\\[init] Loaded config:\n{cfg}")
        Path(f"{os.environ['RUN_DIR']}/code.cfg").write_text(cfg + "\n")

        setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

        self.seed_everything(root_cfg.cfg.seed)
        self.main()

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def main(self) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass
