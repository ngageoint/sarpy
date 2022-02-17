"""
Setup module for SarPy.
"""

import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()


# Get the relevant setup parameters from the package
parameters = {}
with open(os.path.join(here, 'sarpy', '__about__.py'), 'r') as f:
    exec(f.read(), parameters)


def my_package_data():
    def find_dirs(init_dir, start, the_list):
        for root, dirs, files in os.walk(os.path.join(init_dir, start)):
            include_root = False
            for fil in files:
                if os.path.splitext(fil)[1] == '.xsd':
                    include_root = True
                    break
            if include_root:
                root_dir = root.replace('\\', '/')
                rel_dir = root_dir[len(init_dir):]
                if rel_dir[0] == '/':
                    rel_dir = rel_dir[1:]
                if rel_dir[-1] == '/':
                    rel_dir = rel_dir[:-1]
                the_list.append(rel_dir + '/*.xsd')

    package_list = []
    find_dirs('sarpy', 'io/complex/sicd_schema/', package_list)
    find_dirs('sarpy', 'io/phase_history/cphd_schema/', package_list)
    find_dirs('sarpy', 'io/phase_history/crsd_schema/', package_list)
    find_dirs('sarpy', 'io/product/sidd_schema/', package_list)
    find_dirs('sarpy', 'annotation/afrl_rde_schema/', package_list)
    return package_list


def my_test_suite():
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
      include_package_data=True,
      package_data={'sarpy': my_package_data()},
      url=parameters['__url__'],
      author=parameters['__author__'],
      author_email=parameters['__email__'],  # The primary POC
      install_requires=['numpy>=1.11.0', 'scipy'],
      zip_safe=False,  # Use of __file__ and __path__ in some code makes it unusable from zip
      test_suite="setup.my_test_suite",
      tests_require=[],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ],
      platforms=['any'],
      license='MIT')
