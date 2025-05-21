import logging
from typing import Optional

from accelerate import Accelerator
from accelerate.logging import get_logger as _get_accel_logger


def setup_logger(
    name: str = __name__, level: int = logging.INFO, accelerator: Optional[Accelerator] = None
) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    If an Accelerator instance is provided, use its logger.

    Parameters
    ----------
    name : str
        The name of the logger. Default is the module name.
    level : int
        The logging level. Default is logging.INFO.
    accelerator : Optional[Accelerator]
        An optional Accelerator instance. If provided, the logger will be
        set up using the Accelerator's logging system.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    # Use Accelerate's logger if requested and available
    if accelerator is not None:
        # Convert numeric level to string (e.g., 20 -> "INFO")
        level_name = logging.getLevelName(level)
        return _get_accel_logger(name, log_level=level_name)

    # Fallback to standard Python logging
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # If no handlers are set up yet, add a default StreamHandler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
