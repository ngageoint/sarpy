SarPy
=====
SarPy is a basic Python library to read, write, display, and do simple processing
of complex SAR data using the NGA [SICD](http://www.gwg.nga.mil/ntb/baseline/docs/SICD/)
format. It has been released by NGA to encourage the use of SAR data standards
throughout the international SAR community. SarPy complements the
[SIX](https://github.com/ngageoint/six-library) library (C++) and the
[MATLAB SAR Toolbox](https://github.com/ngageoint/MATLAB_SAR), which are
implemented in other languages but have similar goals.

Some sample SICD files can be found 
[here](https://github.com/ngageoint/six-library/wiki/Sample-SICDs).

In addition to SICD, SarPy can also read COSMO-SkyMed, RADARSAT-2, Radar Constellation 
Mission (RCM), and Sentinel-1 SLC formats and convert them to SICD.

Some examples of how to read complex SAR data using SarPy are provided in 
`docs/sarpy_example.py`.

Origins
-------
SarPy was developed at the National Geospatial-Intelligence Agency (NGA). The 
software use, modification, and distribution rights are stipulated within the 
MIT license.

Dependencies
------------
The core library functionality depends only on `numpy >= 1.11.0` with some minor 
dependency on `scipy`. 

Optional Dependencies and Behavior
----------------------------------
There are a small collection of dependencies representing functionality which may 
not be core requirements for much of the sarpy targeted tasks. The tension between
requiring the least extensive list of dependencies possible for core functionality 
and not having surprise unstated dependencies which caused unexpected failures is 
evident here. It is evident that there are many viable arguments for making any 
or all of these formally stated dependencies. The choices made here are guided by 
practical realities versus what is generally considered best practices.

For all packages on this list, the import is tried (where relevant), and any 
import errors fr these optional dependencies are caught and handled. In other words, 
a missing optional dependency **will not** be presented as import time. Excepting 
the functionality requiring `h5py`, this import error handling is probably silent. 

Every module in sarpy can be successfully imported, provided that numpy and scipy 
are in the environment. Attempts at using functionality depending on a missing 
optional dependency will generate an error **at run time** with accompanying 
message indicating the missing optional dependency.

- Support for reading single look complex data from certain sources which provide 
  data in hdf5 format require the `h5py` package, this includes Cosmo-Skymed, ICEYE, 
  and NISAR data.

- Reading an image segment in a NITF file using jpeg or jpeg 2000 compression 
  and/or writing a kmz image overlay requires the `pillow` package.

- CPHD consistency checks, presented in the `sarpy.consistency` module, depend on 
  `lxml>=4.1.1`, `networkx>=2.5`, `shapely>=1.6.4`, and `pytest>=3.3.2`. Note that these
  are the versions tested for compliance.

- Some less commonly used (in the sarpy realm) NITF functionality requires the use 
  and interpretation of UTM coordinates, and this requires the `pyproj` package. 

- Building sphinx documentation (mentioned below) requires packages `sphinx`, 
  `sphinxcontrib-napoleon`, and `sphinx_gallery`.

- Optional portions of running unit tests (unlikely to be of relevance to anyone 
  not performing development on the core sarpy package itself) require the `lxml`
  package

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

Documentation
-------------
Documentation for the project is available at [readthedocs](https://sarpy.readthedocs.io/en/latest/).

If this documentation is inaccessible, it can be built locally after checking out 
this repository using sphinx via the command `python setup.py build_sphinx`. 
This depends on python packages `sphinx` and `sphinxcontrib-napoleon`.

Issues and Bugs
---------------
Support for Python 2 has been dropped.

The core sarpy functionality has been tested for Python 3.6, 3.7, 3.8, and 3.9. 
Other versions should be considered unsupported. Changes to sarpy for the sole 
purpose of supporting a Python version beyond end-of-life are unlikely to be 
considered.

Information regarding any discovered bugs would be greatly appreciated, so please
feel free to create a github issue. If more appropriate, **do not hesitate to 
contact thomas.mccullough.ctr@nga.mil for assistance.**

Pull Requests
-------------
Efforts at direct contribution to the project are certainly welcome, and please
feel free to make a pull request. Note that any and all contributions to this 
project will be released under the MIT license.

Software source code previously released under an open source license and then 
modified by NGA staff is considered a "joint work" (see 17 USC 101); it is partially 
copyrighted, partially public domain, and as a whole is protected by the copyrights 
of the non-government authors and must be released according to the terms of the 
original open source license.

Associated GUI Capabilities moved to individual repositories - June 2020
------------------------------------------------------------------------
In addition to a complete refactor of the core capabilities, graphical user interface
functionality were first introduced in March 2020. In June 2020, these 
capabilities were split out of the sarpy repository into their own repositories 
in the NGA project. See the [sarpy_apps](https://github.com/ngageoint/sarpy_apps), 
which depends on [tk_builder](https://github.com/ngageoint/tk_builder). 
