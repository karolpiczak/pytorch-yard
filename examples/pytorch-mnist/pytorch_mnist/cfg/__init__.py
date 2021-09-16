from dataclasses import dataclass

import pytorch_yard


# Experiment settings validation schema & default values
@dataclass
class Settings(pytorch_yard.Settings):
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    momentum: float = 0.9
