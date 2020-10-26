SarPy
=====
SarPy is a basic Python library to read, write, display, and do simple processing
of complex SAR data using the NGA [SICD](http://www.gwg.nga.mil/ntb/baseline/docs/SICD/)
format. It has been released by NGA to encourage the use of SAR data standards
throughout the international SAR community. SarPy complements the
[SIX](https://github.com/ngageoint/six-library) library (C++) and the
[MATLAB SAR Toolbox](https://github.com/ngageoint/MATLAB_SAR), which are
implemented in other languages but have similar goals.

Some sample SICD files can be found [here](https://github.com/ngageoint/six-library/wiki/Sample-SICDs).

In addition to SICD, SarPy can also read COSMO-SkyMed, RADARSAT-2, Radar Constellation 
Mission (RCM), and Sentinel-1 SLC formats and convert them to SICD.

Some examples of how to read complex SAR data using SarPy are provided in `docs/sarpy_example.py`.

Origins
-------
SarPy was developed at the National Geospatial-Intelligence Agency (NGA). The software use,
modification, and distribution rights are stipulated within the MIT license.

Dependencies
------------
The core library functionality depends only on `numpy >= 1.9.0` with some very minor
dependency on `scipy`. Support for reading Cosmo-Skymed data (contained in hdf5 files)
requires the `h5py` package.

Installation
------------
From PyPI, install using pip (may require escalated privileges e.g. sudo):
```bash
pip install sarpy
```
Note that here `pip` represents the pip utility for the desired Python environment.

From the top level of a cloned version of this repository, install for all users of 
your environment (may require escalated privileges, e.g. sudo):
```bash
python setup.py install
```
Again, `python` here represents the executible associated with the desired Python 
environment.

For more verbose instructions for installing from source, such as how to perform an 
install applicable for your user only and requiring no escalated privileges, 
see [here](https://docs.python.org/3/install/index.html).

Python 2.7
----------
The development here has been geared towards Python 3.6 and above, but efforts have
been made towards remaining compatible with Python 2.7. If you are using the library
from Python 2.7, there is an additional dependency for the `typing` package, easily
installed using conda or pip.

For the new GUI capabilities described below, you additionally need `pillow` and
`matplotlib`, as well as the `future` package (not to be confused with the more
widely known `futures`), all of which can be installed using conda or pip.

Breaking Changes - March 2020
-----------------------------
In March 2020, the sarpy library has undergone a complete refactor, and the
most profound changes have occurred around the particulars of the SICD meta-data
structure (a completely object-oriented approach has been adopted), and some of the
particulars for the file reading and writing.

The intent of this effort is to provide better long-term stability, and additional
capabilities and functionality. Unfortunately, changing the inner workings of the
SICD data structure is almost certain to contain some breaking changes for particular
uses.

**Please do not hesitate to contact thomas.mccullough.ctr@nga.mil for assistance**

Associated GUI Capabilities moved to individual repositories - June 2020
------------------------------------------------------------------------
In addition to a complete refactor of the core capabilities, graphical user interface
functionality were first introduced in March 2020. In June 2020, these 
capabilities were split out of the sarpy repository into their own repositories 
in the NGA project. See the [sarpy_apps](https://github.com/ngageoint/sarpy_apps), 
which depends on [tk_builder](https://github.com/ngageoint/tk_builder). 

Documentation
-------------
The documentation of this project is a work in progress, particularly for the
new GUI efforts. It is the desire for the documentation to be hosted someplace
appropriate in the near future.

Currently, basic documentation can be built after checking out this repository
using sphinx via the command `python setup.py build_sphinx`. This depends
on python packages `sphinx` and `sphinxcontrib-napoleon`.

Issues and Bugs
---------------
The core sarpy functionality has been tested for Python 2.7.17, 3.6, 3.7, and 3.8.
Other versions should be considered unsupported. The new GUI capabilities have been
less extensively tested, and should be considered experimental at this point.

Information regarding any discovered bugs would be greatly appreciated, so please
feel free to create a github issue.

Pull Requests
-------------
Efforts at direct contribution to the project are certainly welcome, and please
feel free to make a pull request. The only caveat is that all contributions to
this project will be released under the MIT license.

Software source code previously released under an open source license and then modified
by NGA staff is considered a "joint work" (see 17 USC 101); it is partially copyrighted,
partially public domain, and as a whole is protected by the copyrights of the non-government
authors and must be released according to the terms of the original open source license.
