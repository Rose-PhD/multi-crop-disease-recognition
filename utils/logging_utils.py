"""
utils/logging_utils.py — Shared logging setup.

Provides
--------
setup_logger : configure a named logger with console + rotating file handlers.

Usage
-----
    from utils import setup_logger
    logger = setup_logger("flat.train", log_file="/path/to/train.log")
    logger.info("Starting fold 1")
"""

import logging
from pathlib import Path


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    """
    Return a logger that writes INFO+ to both the console and *log_file*.

    The log file is created (and its parent directories) if they do not exist.
    Calling setup_logger with the same *name* twice is safe — handlers are
    only added once.

    Parameters
    ----------
    name     : dotted logger name, e.g. "flat.train" or "hier.test"
    log_file : path to the .log file; parent dirs are created automatically.

    Returns
    -------
    logging.Logger
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # already configured — return as-is
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # file handler — DEBUG and above (captures everything)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
