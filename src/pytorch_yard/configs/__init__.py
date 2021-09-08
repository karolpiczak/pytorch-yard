# Using redundant module aliases for public export
# https://github.com/microsoft/pyright/blob/master/docs/typed-libraries.md#library-interface

from .cfg import Settings as Settings
from .register import RootConfig as RootConfig
from .register import get_tags as get_tags
from .register import register_configs as register_configs
