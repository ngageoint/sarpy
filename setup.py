"""
Setup module for SarPy.
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
# from os import path
import os
try:
    # If attempting hard links cause "error removing..." errors (can occur in Windows on network
    # drives), this will fix it:
    del os.link  # TODO: is this still viable?
except AttributeError:
    pass


# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# Get the relevant setup parameters from the package
parameters = {}
with open(os.path.join(here, 'sarpy', '__about__.py'), encoding='utf-8') as f:
    exec(f.read(), parameters)


# try to prepare the sphinx arguments for doc building integeration
try:
    from sphinx.setup_command import BuildDoc
    sphinx_args = {
        'cmdclass': {'build_sphinx': BuildDoc},
        'command_options' : {
        'build_sphinx': {
            'project': ('setup.py', 'sarpy'),
            'version': ('setup.py', parameters['__version__']),
            'release': ('setup.py', parameters['__release__']),
            'copyright': ('setup.py', parameters['__copyright__']),
            'source_dir': ('setup.py', os.path.join(here, 'docs', 'sphinx'))}
        }
    }
except ImportError:
    sphinx_args = {}


setup(name=parameters['__title__'],
      version=parameters['__version__'],
      description=parameters['__summary__'],
      long_description=long_description,
      long_description_content_type='text/x-rst',
      packages=find_packages(),  # Should find SarPy and all subpackages
      package_data={'': ['*.xsd']},  # Schema files are required for parsing SICD
      url=parameters['__url__'],
      author=parameters['__author__'],
      # The primary POC, rather than the author really
      author_email=parameters['__email__'],
      install_requires=['numpy>=1.9.0', 'scipy'],
      extras_require={
        'csk':  ['h5py', ],
        'docs': ['Sphinx', 'sphinxcontrib.napoleon'],
      },
      zip_safe=False,  # Use of __file__ and __path__ in some code makes it unusable from zip
      test_suite="tests",
      # use_scm_version=False,
      # setup_requires=['setuptools_scm'],  # this is probably not used if we set the version?
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      platforms=['any'],
      license='MIT',
      **sphinx_args
      )
