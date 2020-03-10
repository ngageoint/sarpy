SarPy
=====

SarPy is a basic Python library to read, write, display, and do simple processing
of complex SAR data using the NGA `SICD <http://www.gwg.nga.mil/ntb/baseline/docs/SICD/>`_
format. It has been released by NGA to encourage the use of SAR data standards
throughout the international SAR community. SarPy complements the
`SIX <https://github.com/ngageoint/six-library>`_ library (C++) and the
`MATLAB SAR Toolbox <https://github.com/ngageoint/MATLAB_SAR>`_, which are
implemented in other languages but have similar goals.

Some sample SICD files can be found `here <https://github.com/ngageoint/six-library/wiki/Sample-SICDs>`_.

In addition to SICD, SarPy can also read COSMO-SkyMed, RADARSAT-2, Radar Constellation Mission (RCM),
and Sentinel-1 SLC formats and convert them to SICD.

Some examples of how to read complex SAR data using SarPy are provided in `docs/sarpy_example.py`.

Origin
~~~~~~

SarPy was developed at the National Geospatial-Intelligence Agency (NGA). The software use,
modification, and distribution rights are stipulated within the MIT license.

Pull Requests
~~~~~~~~~~~~~

If you would like to contribute to this project, please make a pull request. We will
review the pull request and discuss the changes. All pull request contributions to
this project will be released under the MIT license.

Software source code previously released under an open source license and then modified
by NGA staff is considered a "joint work" (see 17 USC 101); it is partially copyrighted,
partially public domain, and as a whole is protected by the copyrights of the non-government
authors and must be released according to the terms of the original open source license.

Significant Changes - March 2020
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Breaking Changes
----------------
In March 2020, the sarpy library has undergone a complete refactor, and the
most profound changes have occurred around the particulars of the SICD meta-data
structure (a completely object-oriented approach has been adopted), and some of the
particulars for the file reading and writing.

The intent of this effort is to provide better long-term stability, and additional
capabilities and functionality. Unfortunately, changing the inner workings of the
SICD data structure is almost certain to contain some breaking changes for particular
uses.

**Please do not hesitate to contact thomas.mccullough.ctr@nga.mil for assistance**

GUI Capabilities
----------------
In addition to a complete refactor of the core capabilities, graphical user interface
functionality has been introduced. The goal for this capability is to encourage
fast and simple prototyping to enable research. Most notably, some of the most commonly
used TASER capabilities from the MATLAB SAR Toolbox have been recreated using these GUI
components.

The decision was made to use `tkinter <https://docs.python.org/3/library/tkinter.html>`_
for this capability. The particulars of this choice are entirely pragmatic. Most
importantly, `tkinter` is well supported for essentially every architecture that
Python is supported, and there are no complicating factors in licensing, configuration,
or installation. For better or for worse, `tkinter` will work on essentially any
government system with a viable Python environment right out of the box. The same
cannot generally be said for the other popular GUI frameworks like QT, WX, or GTK.

Documentation
-------------
The documentation of this project is a work in progress, particularly for the
new GUI efforts. It is the desire for the documentation to be hosted someplace
appropriate in the near future.

Currently, basic documentation can be built after checking out this repository
using sphinx via the command :code:`python setup.py build_sphinx`. This depends
on python packages `sphinx` and `sphinxcontrib-napoleon`.
