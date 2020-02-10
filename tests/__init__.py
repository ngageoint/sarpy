import sys
import logging

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

logging.basicConfig(level=logging.INFO)

