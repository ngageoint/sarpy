"""Defines logging customization for DEM class usage"""

import logging

_DEFAULT_MESSAGE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def dem_logger(name,
               logfile=None,
               level=None,
               log_to_console=True,
               message_format=_DEFAULT_MESSAGE_FORMAT, date_format=_DEFAULT_DATE_FORMAT):
    """
    Configure logger handling for DEM class.

    :param name: name/id for logger instance
    :param logfile: file name for logfile. Empty string for not logging to file.
    :param level: logging level, either string enum for logging level, or integer value (i.e. logging.<LEVEL>).
        Will default to logging.INFO.
    :param log_to_console: boolean for whether to log to the console (in addition to any file logging)
    :return: logger instance
    """

    default_formatter = logging.Formatter(fmt=message_format, datefmt=date_format)  # Set formatter

    logger = logging.getLogger(name)  # Get/define the logger instance
    console_handler = logging.StreamHandler()  # Get a stream handler instance

    if logfile is not None:  # If logging to a file set up and add the file handler
        file_handler = logging.FileHandler(logfile, mode='a')  # Get the file handler
        file_handler.setFormatter(default_formatter)  # Set formatter
        file_handler.setLevel(logging.DEBUG)  # We ALWAYS want to capture ALL messages in the log file
        if 'FileHandler' not in ''.join(str(logger.handlers)):
            # TODO: MEDIUM - this is bad type checking
            logger.addHandler(file_handler)  # Add the file handler to the logger

    # Set default and console logging levels
    default_level = logging.WARNING
    log_level = None
    if level is None:
        log_level = default_level
    elif isinstance(level, str):
        log_level = getattr(logging, level.upper(), None)
    elif isinstance(level, int):
        log_level = level  # this could be a dumb value, and setting may fail

    try:
        logger.setLevel(log_level)
        console_handler.setLevel(log_level)
    except ValueError:
        logging.WARNING('Setting logging level to {} failed. Setting to logging.WARNING.'.format(level))
        logger.setLevel(default_level)
        console_handler.setLevel(default_level)

    if log_to_console:  # If we want to print to the console/terminal
        console_handler.setFormatter(default_formatter)
        if 'StreamHandler' not in ''.join(str(logger.handlers)):  # Ensures just one console message
            # TODO: MEDIUM - this kind of type checking is not good
            logger.addHandler(console_handler)

    return logger
