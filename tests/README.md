Relevance of these unit tests to users:
=======================================
There is no configuration of sarpy beyond establishing a python environment with
appropriate dependencies.

**This suite of unit tests are really only relevant to developers adding or modifying
core capabilities of sarpy.** Users can otherwise safely ignore these tests.


Unit Test Approach and Configuration:
=====================================
The traditional unit test approach aims for small capability checks which are ideally
self-contained tests with no dependencies on anything external to the test itself.
If a test does have any dependencies, it is ideal that any dependency is distributed
with the test itself (i.e. in the same git repository).

The most important sarpy capabilities are centered around reading complex imagery
in a wide variety of different formats. Any tests for the most important capabilities
of sarpy inherently depend on files and packages in a variety of formats. For a variety
of reasons, this means that effective tests depend on files much too large to be
feasible distributed with git, and some files cannot be freely distributed.

A comprehensive suite of test files will be available to developers with an
appropriate relationship with NGA. Many developers may only be interested in
testing reading capability for a single format, and could easily identify their
own collection of test files.

File Identification:
--------------------
Files used for individual tests will be identified is a json structure referenced
in a given unit test. Tests referencing paths that don't exist will not be performed,
so you only need to assemble or reference files that you care to test.

There are two cases for paths provided for these tests:
1. The path is identified as absolute (the default). User path lengthening will be
   performed, so the path `'~/'` will evaluate to your home directory.
2. The path is identified as relative, all such paths will be assumed relative to
   the same parent, and environment variable `SARPY_TEST_PATH` must be defined as
   this **absolute** parent path. If this environment variable is not set, then an
   error will result.

Run the Tests:
--------------
Unit tests can be run using the command `python setup.py test` or `pytest`, if
using pytest.

Testing across multiple environments
------------------------------------
Prerequisites:
- [conda package manager](https://docs.conda.io/en/latest/)
- directory containing SarPy test files

Steps:
```shell
conda create -n sarpytest python --y
conda activate sarpytest

pip install nox
export SARPY_TEST_PATH=/data/sarpy_test  # set to appropriate dir
nox
```

File Sources:
=============

Non-SAR data:
-------------
- General NITF files:
    - https://gwg.nga.mil/gwg/focus-groups/ntb/jbp-sample-files/
    - The relevant files here are the NITF files with extension .ntf. There are
      non-NITF (.nsf) files here that are not presently relevant to sarpy.
    - **These files are referenced for testing general nitf header parsing capabilities.**
      None of the sample NITF files here appear to be SAR related files.

- Geoid data:
    - Overall found here, specifics below - https://geographiclib.sourceforge.io/html/geoid.html
    - Test data - https://sourceforge.net/projects/geographiclib/files/testdata/GeoidHeights.dat.gz
    - EGM96
        - 5 minute data:
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-5.tar.bz2
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-5.zip
        - 15 minute data:
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-15.tar.bz2
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm96-15.zip
    - EGM2008
        - 1 minute data:
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-1.tar.bz2
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-1.zip
        - 5 minute data:
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-5.tar.bz2
            + https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-5.zip

- DTED:
    - Shuttle Radar Topography Mission (SRTM) data from https://earthexplorer.usgs.gov

Publicly available complex valued SAR example data in various formats:
----------------------------------------------------------------------
- SICD (a NITF file following particular rules):
    + https://github.com/ngageoint/six-library/wiki/Sample-SICDs
    + These files should be referenced in unit tests for general NITF header parsing,
      and extensive tests of SICD reading and writing capabilities.
- Radarsat-2:
    + https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/
    + Only the SLC (single-look complex) product is presently relevant to sarpy.
- RCM (Radarsat Constellation Mission):
    + https://www.asc-csa.gc.ca/eng/open-data/access-the-data.asp
    + Only the SLC (single-look complex) product is presently relevant to sarpy.
- Sentinel-1:
    + https://search.asf.alaska.edu/#/?resultsLoaded=false&zoom=3&center=-97.493959,39.672786&view=equitorial&productTypes=SLC
    + The whole Sentinel catalog seems available here.
- TerraSar-X:
    + https://www.intelligence-airbusds.com/en/8262-sample-imagery?type=364
    + Only the SSC product is presently relevant to sarpy.
- ICEYE:
    + https://www.iceye.com/downloads/datasets
    + Only the SSC product is presently relevant to sarpy.
- PALSAR:
    + https://search.asf.alaska.edu/#/?resultsLoaded=true&zoom=3&center=-97.493959,39.672786&view=equitorial&dataset=ALOS&productTypes=L1.1
    + Only the complex valued data is relevant to sarpy.
- Capella
    + https://capella-open-data.s3.us-west-2.amazonaws.com/
    + Only the SLC (single-look complex) product is presently relevant to sarpy.
- NISAR
    + https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=sabine_01200_17088_017_170901_L090_CX_05#data
    + Simulated NISAR product.
- CPHD
    + https://rdr.ucl.ac.uk/articles/dataset/T-72_SAR_data_dome_Remote_Sensing_0_25_no_PSF_/20681836
    + Simulated CPHD files.

Not publicly available complex SAR data:
----------------------------------------
- Cosmo Skymed
- KompSat - very similar format as Cosmo Skymed, and read by the same reader.
