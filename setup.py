# -*- coding: utf-8 -*-
"""
Setup module for SarPy.
"""

import sys
from setuptools import setup, find_packages
from codecs import open

import os
try:
    # If attempting hard links cause "error removing..." errors,
    # which can occur in Windows on network drives.
    # This may fix it, but may be deprecated?
    del os.link
except AttributeError:
    pass


# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()


# Get the relevant setup parameters from the package
parameters = {}
with open(os.path.join(here, 'sarpy', '__about__.py'), 'r') as f:
    exec(f.read(), parameters)


install_requires = ['numpy>=1.11.0', 'scipy']
tests_require = []
if sys.version_info[0] < 3:
    tests_require.append('unittest2')
    # unittest2 only for Python2.7, we rely on subTest usage
    install_requires.append('typing')

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
      package_data={'sarpy': ['*.xsd']},  # Schema files for SICD standard(s)
      url=parameters['__url__'],
      author=parameters['__author__'],
      author_email=parameters['__email__'],  # The primary POC
      install_requires=install_requires,
      extras_require={
        'csk':  ['h5py', ],
      },
      zip_safe=False,  # Use of __file__ and __path__ in some code makes it unusable from zip
      test_suite="setup.my_test_suite",
      tests_require=tests_require,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      platforms=['any'],
      license='MIT')
