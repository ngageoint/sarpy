"""
THIS MODULE SETS UP LOGGING FOR DEMPY
"""

import os
import logging

def dem_logger(name, logfile='', level='DEBUG', log_to_console=True):
    '''
    Set up and return a formatted logger
    '''
    message_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Specifies the output message style
    date_format = '%Y-%m-%d %H:%M:%S'  # Specifies the date style
    default_formatter = logging.Formatter(fmt=message_format, datefmt=date_format)  # Set formatter

    logger = logging.getLogger(name) # Get the logger
    console_handler = logging.StreamHandler()  # Get the stream handler, even if we're not going to use it

    if not logfile == '':  # If logging to a file set up and add the file handler
        file_handler = logging.FileHandler(logfile,mode='a')  # Get the file handler
        file_handler.setFormatter(default_formatter)  # Set formatter
        file_handler.setLevel(logging.DEBUG)  # We ALWAYS want to capture ALL messages in the log file
        if 'FileHandler' not in ''.join(str(logger.handlers)):
            logger.addHandler(file_handler)  # Add the file handler to the logger

    # Set default and console logging levels
    level = level.upper() # Just in case a lowercase level is given to avoid errors
    if level == 'NOTSET':
        logger.setLevel(logging.NOTSET)
        console_handler.setLevel(logging.NOTSET)
    elif level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
        console_handler.setLevel(logging.WARNING)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)
        console_handler.setLevel(logging.ERROR)
    elif level == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)
        console_handler.setLevel(logging.CRITICAL)
    elif level == 'FATAL':
        logger.setLevel(logging.FATAL)
        console_handler.setLevel(logging.FATAL)
    else:
        logging.WARNING('UNRECOGNIZED LOGGING LEVEL')

    if log_to_console:  # If we want to print to the console/terminal
        console_handler.setFormatter(default_formatter)
        if 'StreamHandler' not in ''.join(str(logger.handlers)):  # Ensures just one console message
            logger.addHandler(console_handler)

    return logger

# END OF FILE