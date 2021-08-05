from .__about__ import *
import logging


__all__ = ['__version__',
           '__classification__', '__author__', '__url__', '__email__',
           '__title__', '__summary__',
           '__license__', '__copyright__']


# establish logging paradigm
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
