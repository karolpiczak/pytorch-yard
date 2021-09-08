from dataclasses import dataclass
from typing import List, Optional


# Experiment settings validation schema & default values
@dataclass
class Settings:
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    # wandb tags
    _tags_: Optional[List[str]] = None

    # Seed for all random number generators
    seed: int = 1
