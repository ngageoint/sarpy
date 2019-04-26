"""
Setup module for SarPy.
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
# from os import path
import os
# If attempting hard links cause "error removing..." errors (can occur in Windows on network
# drives), this will fix it:
del os.link

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='sarpy',
      description='Reading/writing/simple processing of complex SAR data in Python via SICD',
      long_description=long_description,
      long_description_content_type='text/x-rst',  # Default but "Explicit is better than implicit"
      packages=find_packages(),  # Should find SarPy and all subpackages
      package_data={'': ['*.xsd']},  # Schema files are required for parsing SICD
      url='https://github.com/ngageoint/sarpy',
      author='National Geospatial-Intelligence Agency',
      # Not the only author, but currently the primary POC
      author_email='Wade.C.Schwartzkopf.ctr@nga.mil',
      # Many portions of sarpy run fine without scipy, and we have tried to avoid this dependency
      # wherever possible, but there are enough of these dependencies scattered throughout that we
      # declare it here.
      install_requires=['numpy', 'scipy'],
      extras_require={
        'csk':  ['h5py'],
      },
      zip_safe=False,  # Use of __file__ and __path__ in some code makes it unusuable from zip
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      # python_requires is really just the NumPy requirement, so maybe we don't need to state
      # python_requires explicitly as it is already implicitly declared in dependency stated above
      # python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      platforms=['any'],
      license='MIT'
      )
