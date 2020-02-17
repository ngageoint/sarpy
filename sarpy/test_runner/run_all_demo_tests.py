import unittest
import os

loader = unittest.TestLoader()
this_dir = os.path.dirname(os.path.realpath(__file__))
start_dir = os.path.join(this_dir, "demo_tests")
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
