
import logging
import os
from logging import FileHandler, LogRecord, StreamHandler

from rich.console import Console
from rich.text import Text


class RichConsoleHandler(StreamHandler):  # type: ignore
    """
    Logging handler for rich console output.
    """

    def __init__(self) -> None:
        super().__init__()  # type: ignore

        self.console = Console()

    def emit(self, record: LogRecord) -> None:
        try:
            self.console.print(self.format(record))  # type: ignore
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class RichFileHandler(FileHandler):
    """
    Logging handler for file output with stripped formatting. 
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)

    def emit(self, record: LogRecord) -> None:
        record.msg = Text.from_markup(str(record.msg)).plain
        return super().emit(record)


LOG_LEVEL = logging.DEBUG if os.getenv('RUN_MODE', '').lower() == 'debug' else logging.INFO

log = logging.getLogger('main')
log.setLevel(LOG_LEVEL)


def debug(msg: str):
    log.debug(f':information_source: {msg}')


def info(msg: str):
    log.info(f'{msg}')


def info_bold(msg: str):
    log.info(f'[bold yellow]{msg}')


def warning(msg: str):
    log.warning(f':exclamation:[bold yellow]{msg}')


def error(msg: str):
    log.error(f':x: [bold red]{msg}')
