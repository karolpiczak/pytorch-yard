import torch

from ..core import Experiment


class PyTorchExperiment(Experiment):
    def before_entry(self) -> None:
        self.device: torch.device

    def main(self) -> None:
        pass

    def finish(self) -> None:
        pass
