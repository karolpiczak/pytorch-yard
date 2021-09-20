from typing import Optional, Type, cast

import pytorch_yard
import torch
import torchvision.transforms as transforms
from pytorch_yard.experiments.pytorch import PyTorchExperiment
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_mnist.cfg import Settings


class Experiment(PyTorchExperiment):

    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        super().__init__(config_path, settings_cls, settings_group=settings_group)

        self.cfg: Settings
        """ Experiment config. """

    def entry(self, root_cfg: pytorch_yard.RootConfig):
        super().entry(root_cfg)

    def main(self) -> None:
        # ------------------------------------------------------------------------------------------
        # Init
        # ------------------------------------------------------------------------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ------------------------------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------------------------------
        # mnist = MNIST(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        # mnist_transforms = transforms.Compose([
        #     transforms.Pad(2, fill=0, padding_mode='constant'),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        #     transforms.Normalize(mean=(0.1000,), std=(0.2752,)),
        # ])


if __name__ == '__main__':
    Experiment('pytorch_mnist', Settings)
