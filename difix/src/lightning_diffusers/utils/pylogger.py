import logging
from typing import Optional

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name: Optional[str] = None) -> logging.Logger:


    logger = logging.getLogger(name)


    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


class RankedLogger:


    def __init__(self, name: Optional[str] = None, rank_zero_only: bool = False) -> None:

        self._logger = logging.getLogger(name)
        self._rank_zero_only = rank_zero_only

    def debug(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.debug)(message, *args, **kwargs)
        else:
            self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.info)(message, *args, **kwargs)
        else:
            self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.warning)(message, *args, **kwargs)
        else:
            self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.error)(message, *args, **kwargs)
        else:
            self._logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.exception)(message, *args, **kwargs)
        else:
            self._logger.exception(message, *args, **kwargs)

    def fatal(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.fatal)(message, *args, **kwargs)
        else:
            self._logger.fatal(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        if self._rank_zero_only:
            rank_zero_only(self._logger.critical)(message, *args, **kwargs)
        else:
            self._logger.critical(message, *args, **kwargs)
