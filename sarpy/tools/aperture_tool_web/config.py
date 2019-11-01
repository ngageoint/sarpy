import logging
import os
import sys

from logging import Formatter
from logging.handlers import RotatingFileHandler


SECRET_KEY = os.environ['FLASK_SECRET_KEY']
ATK_PATH = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.environ['ATK_API_KEY']

sys.path.append(ATK_PATH)
dirname = os.path.dirname
logfile = os.path.join(dirname(__file__), 'logs', 'app.log')
handler = RotatingFileHandler(logfile, maxBytes=10240, backupCount=10)
handler.setLevel(logging.DEBUG)
handler.setFormatter(Formatter(
    '%(asctime)s %(levelname)s %(message)s [in %(pathname)s:%(lineno)d]'
))
# Add additional logger handlers here, add to LOG_HANDLERS list

LOG_HANDLERS = [handler, ]

# Add your custom settings below
CHAIN_HISTORY_LENGTH = 1
