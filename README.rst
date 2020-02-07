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

Some examples of how to read complex SAR data using SarPy are provided in docs/sarpy_example.py.

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
by NGA staff is considered a "joint work" (see 17 USC § 101); it is partially copyrighted,
partially public domain, and as a whole is protected by the copyrights of the non-government
authors and must be released according to the terms of the original open source license.

Documentation
~~~~~~~~~~~~~

The documentation of this project is a work in progress. Currently, basic documentation
can be built using sphinx via the command :code:`python setup.py build_sphinx`. This depends
on python packages Sphinx and sphinxcontrib-napoleon.
