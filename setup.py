# -*- coding: utf-8 -*-
"""
Setup module for SarPy.
"""

import os
import sys
from setuptools import setup, find_packages
from codecs import open



# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()


# Get the relevant setup parameters from the package
parameters = {}
with open(os.path.join(here, 'sarpy', '__about__.py'), 'r') as f:
    exec(f.read(), parameters)


def my_test_suite():
    if sys.version_info[0] < 3:
        import unittest2 as unittest
    else:
        import unittest
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', top_level_dir='.')
    return test_suite

setup(name=parameters['__title__'],
      version=parameters['__version__'],
      description=parameters['__summary__'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=('*tests*', )),
      package_data={'sarpy': ['io/complex/sicd_schema/*.xsd',
                              'io/phase_history/cphd_schema/*.xsd', ]},
      url=parameters['__url__'],
      author=parameters['__author__'],
      author_email=parameters['__email__'],  # The primary POC
      install_requires=['numpy>=1.11.0', 'scipy', 'typing;python_version<"3.4"'],
      zip_safe=False,  # Use of __file__ and __path__ in some code makes it unusable from zip
      test_suite="setup.my_test_suite",
      tests_require=["unittest2;python_version<'3.4'", ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ],
      platforms=['any'],
      license='MIT')
