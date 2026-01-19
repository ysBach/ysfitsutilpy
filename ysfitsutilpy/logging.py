"""Logging configuration for ysfitsutilpy.

This module provides a centralized logging infrastructure for ysfitsutilpy,
following Python library best practices.

By default, the logger uses a NullHandler to be silent (library mode).
Users can enable console logging for interactive use.

Examples
--------
>>> import ysfitsutilpy as yfu
>>> import logging
>>> # Enable INFO-level logging for interactive use
>>> yfu.enable_console_logging(level=logging.INFO)
>>> # Or use DEBUG for detailed diagnostics
>>> yfu.enable_console_logging(level=logging.DEBUG)
>>> # Standard logging configuration also works
>>> logging.getLogger("ysfitsutilpy").setLevel(logging.WARNING)
"""
import logging

__all__ = ["logger", "set_log_level", "enable_console_logging"]

# Package-level logger using standard naming convention
# When imported as a package, this will be 'ysfitsutilpy'
logger = logging.getLogger("ysfitsutilpy")

# Default: NullHandler (silent by default for library usage)
# This follows the Python logging best practices for libraries:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logger.addHandler(logging.NullHandler())


def set_log_level(level):
    """Set the log level for ysfitsutilpy.

    Parameters
    ----------
    level : int or str
        Logging level. Can be an integer (e.g., logging.DEBUG, logging.INFO)
        or a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Examples
    --------
    >>> import ysfitsutilpy as yfu
    >>> import logging
    >>> yfu.set_log_level(logging.DEBUG)
    >>> yfu.set_log_level('INFO')  # Also accepts strings
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


def enable_console_logging(level=logging.INFO, format=None):
    """Enable console logging for interactive use.

    This is a convenience function to quickly enable visible logging output
    when using ysfitsutilpy interactively. By default, the library is silent
    (only a NullHandler is attached).

    Parameters
    ----------
    level : int, optional
        Logging level. Default: logging.INFO.
        Common values: logging.DEBUG (10), logging.INFO (20),
        logging.WARNING (30), logging.ERROR (40).
    format : str, optional
        Log format string. Default: "[%(levelname)s] %(message)s".
        Set to a custom format string to include timestamps, etc.
        Example: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    Examples
    --------
    >>> import ysfitsutilpy as yfu
    >>> import logging
    >>> # Basic usage - INFO level
    >>> yfu.enable_console_logging()
    >>> # Detailed debugging
    >>> yfu.enable_console_logging(level=logging.DEBUG)
    >>> # Custom format with timestamps
    >>> yfu.enable_console_logging(
    ...     level=logging.DEBUG,
    ...     format="%(asctime)s [%(levelname)s] %(message)s"
    ... )
    """
    if format is None:
        format = "[%(levelname)s] %(message)s"

    # Remove any existing StreamHandlers to avoid duplicate output
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(level)
