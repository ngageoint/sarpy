SarPy
=====
SarPy is a basic Python library to read, write, and do simple processing
of complex SAR data using the NGA SICD format *(standards linked below)*. 
It has been released by NGA to encourage the use of SAR data standards
throughout the international SAR community. SarPy complements the
[SIX](https://github.com/ngageoint/six-library) library (C++) and the
[MATLAB SAR Toolbox](https://github.com/ngageoint/MATLAB_SAR), which are
implemented in other languages but have similar goals.

Some sample SICD files can be found 
[here](https://github.com/ngageoint/six-library/wiki/Sample-SICDs).

Relevant Standards Documents
----------------------------
A variety of SAR format standard are mentioned throughout this ReadMe, here are 
associated references. 

*Sensor Independent Complex Data (SICD)* - latest version (1.2.1; 2018-12-13) 
1. [Volume 1, Design & Implementation Description Document](https://nsgreg.nga.mil/doc/view?i=4900)
2. [Volume 2, File Format Description Document](https://nsgreg.nga.mil/doc/view?i=4901)
3. [Volume 3, Image Projections Description Document](https://nsgreg.nga.mil/doc/view?i=4902)
4. [Schema](https://nsgreg.nga.mil/doc/view?i=5230)

*Sensor Independent Derived Data (SIDD)* - latest version (2.0; 2019-05-31)
1. [Volume 1, Design and Implementation Description Document](https://nsgreg.nga.mil/doc/view?i=5009)
2. [Volume 2, NITF File Format Description Document](https://nsgreg.nga.mil/doc/view?i=5016)
3. [Volume 3, GeoTIFF File Format Description Document](https://nsgreg.nga.mil/doc/view?i=5017)
4. [Schema](https://nsgreg.nga.mil/doc/view?i=5231)

*Compensated Phase History Data (CPHD)* - latest version (1.0.1; 2018-05-21)
1. [Design & Implementation Description](https://nsgreg.nga.mil/doc/view?i=4638)
2. [Design & Implementation Schema](https://nsgreg.nga.mil/doc/view?i=4639)

Both SICD and SIDD files are NITF files following specific guidelines
*National Imagery Transmission Format (NITF)* - latest version (2.1, Revision C; 2017-06-06)
1. [National Imagery Transmission Format](https://nsgreg.nga.mil/doc/view?i=4324)

For other NGA standards inquiries, the standards registry can be searched
 [here](https://nsgreg.nga.mil/registries/search/index.jsp?registryType=doc).
 
Basic Capability
----------------
The basic capabilities provided in SarPy is generally SAR specific, and largely 
geared towards reading and manipulating data provided in NGA SAR file formats. 
Full support for reading and writing SICD, SIDD, CPHD, and CRSD (standard pending) 
and associated metadata structures is currently provided, and this is the main 
focus of this project.

There is additionally support for reading data from complex data formats analogous 
to SICD format, *usually called Single Look Complex (SLC) or Level 1*, from a 
variety of commercial or other sources including 
- Capella (**partial support**)
- COSMO-SkyMed (1st and 2nd generation)
- GFF (Sandia format)
- ICEYE
- NISAR
- PALSAR2
- RadarSat-2
- Radar Constellation Mission (RCM)
- Sentinel-1
- TerraSAR-X.

For this SLC format data, it is read directly as though it were coming from a SICD 
file. *This ability to read does not generally apply to data products other 
than the SLC or Level 1 product, and there is typically no direct NGA standard 
analog for these products.*

Some general TIFF and NITF reading support is provided, but this is not the main 
goal of the SarPy library.

Documentation
-------------
Documentation for the project is available at 
[readthedocs](https://sarpy.readthedocs.io/en/latest/).

If this documentation is inaccessible, it can be built locally after checking out 
this repository using sphinx via the command `python setup.py build_sphinx`. 
This depends on python packages `sphinx` and `sphinxcontrib-napoleon`.

Origins
-------
SarPy was developed at the National Geospatial-Intelligence Agency (NGA). The 
software use, modification, and distribution rights are stipulated within the 
MIT license.

Dependencies
------------
The core library functionality depends only on `numpy >= 1.11.0` and `scipy`. 

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

For verbose instructions for installing from source, see 
[here](https://docs.python.org/3/install/index.html). It is recommended that 
still the package is built locally and installed using pip, which allows a proper 
package update mechanism, while `python setup.py install` **does not**.

Issues and Bugs
---------------
Support for Python 2 has been dropped. The core sarpy functionality has been 
tested for Python 3.6, 3.7, 3.8, 3.9, and 3.10. 

Changes to sarpy for the sole purpose of supporting a Python version beyond 
end-of-life are unlikely to be considered.

Information regarding any discovered bugs would be greatly appreciated, so please
feel free to create a github issue. If more appropriate, contact wade.c.schwartzkopf@nga.mil.

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

Associated GUI Capabilities
---------------------------
Some associated SAR specific graphical user interface tools are maintained in the 
[sarpy_apps project](https://github.com/ngageoint/sarpy_apps). 
