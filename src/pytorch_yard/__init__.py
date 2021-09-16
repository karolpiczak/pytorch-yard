__version__ = '0.0.4'

from .configs import RootConfig as RootConfig
from .configs import Settings as Settings
from .core import AvalancheExperiment as AvalancheExperiment
from .core import LightningExperiment as LightningExperiment
from .core import PyTorchExperiment as PyTorchExperiment
from .utils.logging import debug as debug
from .utils.logging import error as error
from .utils.logging import info as info
from .utils.logging import info_bold as info_bold
from .utils.logging import warning as warning
