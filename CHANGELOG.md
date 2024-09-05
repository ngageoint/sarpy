# Change Log

SarPy follows a continuous release process, so there are fairly frequent releases. 
Since essentially every (squash merge) commit corresponds to a release, specific 
release points are not being annotated in GitHub.

## [1.3.59-rc]
### Added
- `noxfile.py`
- TOA visualization to `sarpy/visualization/cphd_kmz_product_creation.py`
- Unit tests for `sarpy/tests/io/complex/sicd_elements`
- `sarpy/tests/io/phase_history/cphd1_elements/conftest.py`
- Unit tests for `sarpy/tests/io/phase_history/cphd1_elements`
- CPHD 1.0.1 xml examples to `sarpy/tests/data`
- `sarpy/processing/sicd/spectral_taper.py` and `sarpy/utils/sicd_sidelobe_control.py`
- `--remap` argument to `sarpy/utils/create_product.py`
- `GDM` to `sarpy/visualization/remap.py`
- Unit tests for `sarpy/consistency/sicd_consistency.py`
- Support reading CPHDs with an AmpSF PVP whose Data/SignalArrayFormat is CF8
- Unit tests for `sarpy/consistency/sidd_consistency.py`
- Support for MATESA TRE
- Support reading/writing CPHDs with compressed signal arrays
- Support for numpy 2.0
### Fixed
- `sarpy.io.kml.add_polygon` coordinate conditioning for older numpy versions
- Replace unsupported `pillow` constant `Image.ANTIALIAS` with `Image.LANCZOS`
- `sarpy/io/phase_history/cphd1_elements/GeoInfo.py` setters
- SquintAngle calculation in `sarpy/io/complex/sicd_elements/SCPCOA.py`
- Incorrectly assigned Graze in SIDD 2.0.0 and SIDD 3.0.0 ExploitationFeatures
- SIDD `TimeCOAPoly` calculation
- Set SIDD Display/Interpolation/Operation values to CORRELATION
- `sarpy.io.phase_history.cphd1_elements.PVP.PVPType.get_size()`
- SICD file reading in `sarpy/consistency/sicd_consistency.py`
- Protect waveform validation from `waveform.TxFreqStart == None` in `sarpy/io/complex/sicd_elements/RadarCollection.py`
- Fix `sarpy/io/complex/sicd_elements/Timeline.py` validation code to allow IPP T1End == T2Start
- Properly close file objects in NITF and CPHD writers
- SIDD file reading in `sarpy/consistency/sidd_consistency.py`
- Application of adjustable parameter offsets in RIC frames during projection
- Overflow bug in `ComplexFormatFunction` magnitude/phase -> real/imag
- NITF image subheader parsing when there are more than 9 bands
- Population of SIDD ExploitationFeatures resolution metadata when processed from a SICD
- Fix BANDSB implementation to parse correctly
- SingleLUTFormatFunction application for LUT with more than one dimension
- SIDD NITF IALVL/IDLVL for NITFs consisting of multiple image segments and/or product images
- Reading/writing of uncompressed NITF image segments with two complex-component bands interleaved by block/row
- Replace deprecated `matplotlib.cm.get_cmap` with `matplotlib.pyplot.get_cmap`
- ReferencePoint attribute in SIDD MeasurementType objects
- Sentinel clutter radiometric polynomials are no longer transposed
- NITFWriter no longer closes passed in file objects
- Various SICD and SIDD elements fixed to better match NGA standard
- Fixed case where DTEDInterpolator applied geoid offset incorrectly
- Improved DTEDInterpolator handling of missing DEMs
- SIDD 3.0.0 point projection
- Restored missing antenna beam footprints in some KMZs
- DTED parsing for tiles with null values
- Improved mapping of SICD -> SIDD polarizations
- Incorrect SIDD ISM.compliesWith definition
- SIO reading/writing
- SIDD 2.0+ FilterType handling

## [1.3.58] - 2023-08-07
### Added
- Added additional tests to test_sicd_elements_geodata.py
### Fixed
- setGeoInfo() definition and typos in GeoData.py.

## [1.3.57] - 2023-08-02
### Added
- Added ability to generate KMZ files from CRSD data.
- Added antenna visualization to complex-image KMZ.
### Changed
- Changed polygon conditioning for KMZs created around antimeridian.
### Fixed
- SARPY correctly using fieldnames to ensure PVPs are transcribed in proper order in
sarpy.io.phase_history.cphd's write_pvp_array function.
- SARPY correctly creating overview for SICDS that have Area but no Area.Plane.

## [1.3.56] - 2023-07-25
### Added
- Added test for SARPY RMA in sicd_elements module. 

## [1.3.55] - 2023-07-19
### Added
- Added test for SARPY to/from XML logic.
### Fixed
- Fixed AntGPid typo to AntGPId.
- Fixed SARPY correctly finding TxAntenna/RcvAntenna PVPs using from_file in cphd_consistency.py.
- Fixed SARPY correctly specifying EBFreqShiftSF in AntPatternType.
- Fixed SARPY correctly handling AddedParameters as a single element with repeated Parameter children.
- Fixed EndPoint type to Endpoint.
- Fixed SARPY correctly including index as a child element instead of an attribute in LSVertexType. 

## [1.3.54] - 2023-07-19
### Added
- Added RadarModeType class to SIDD with SCANSAR option.

## [1.3.53] - 2023-07-11
### Added
- Added 3db contours to transmit and receive beam footprints for CPHD KMZs.

## [1.3.52] - 2023-07-10
### Added
- Added additional tests for sicd_elements ImageCreation.py.

## [1.3.51] - 2023-07-10
### Added
- Added additional tests for sicd_elements CollectionInfo.py.
### Fixed
- Fixed bug in SARPY correctly checking mode types. 

## [1.3.50] - 2023-07-06
### Added
- Added additional tests for sicd_elements Antenna.py.

## [1.3.49] - 2023-07-05
### Fixed
- Fixed bug in SARPY correctly populating NITF FL field.
- Fixed bug in SARPY correctly using context manager interface when using SIDD product
creators to close/flush files.

## [1.3.48] - 2023-07-05
### Added
- Added tests for sicd_elements.blocks.
### Changed
- Changed some string formatting, and value error text in sicd_elements/blocks.py. 

## [1.3.47] - 2023-06-29
### Added
- Added support for SIDD v3.

## [1.3.46] - 2023-06-15
### Added
- Added consistency checks to the SICD validation_checks.py.

## [1.3.45] - 2023-06-15
### Added
- Added tests for sicd_elements module in test_sicd_elements_position.py

## [1.3.44] - 2023-06-14
### Added
- Added ability to generate KMZ files from CPHD data.

## [1.3.43] - 2023-06-09
### Fixed
- Fixed bug in Sarpy in correctly performing approximate value comparisons in consistency.py.

## [1.3.42] - 2023-06-02
### Fixed
- Fixed bug in Sarpy in correctly logging expected and populated value in SPCPCOA.py.
### Added
- Added new tests in test_sicd_elements_scpcoa.py.

## [1.3.41] - 2023-06-02
### Fixed
- Fixed bug in Sarpy not correctly validating schema versions.
### Changed
- Changed 2 term versioning to 3 term versioning for sicd, cphd, sidd, crsd schemas. 
- Changed schema version dates for 1.3.0 sicd and cphd.
- Changed test_sicd_elements_sicd.py to test for version 1.3.0. 

## [1.3.40] - 2023-05-26
### Fixed
- Fixed bug in Sarpy correctly not running a test if there is no geoid file, 
in test_geotiff1deg_reader.py 

## [1.3.39] - 2023-05-26
### Added
- Added new tests in test_sicd_elements_imageformation.py
### Fixed
- Fixed bug in Sarpy correctly setting self.TxFrequencyProc to TxFrequencyProcType when it 
doesn't exist.

## [1.3.38] - 2023-05-23
### Added
- Added llh parser to parsers.py.
### Changed
- Changed readme to align with changes to removal of sphinxcontrib-napoleon dependency.
- Changed dependencies in setup.py to align with readme. 

## [1.3.37] - 2023-05-16
### Added
- Added tests for sicd_elements for SICD.py and RadarCollection.py.
### Changed
- test_sicd_elements.py was split up into smaller more easily manageable python files.
### Fixed
- Fixed bug in SARPY correctly specifying SCP when instantiating GeoDataType in SICD.py
- Fixed spelling error for override parameter in SICD.py, base.py, SIDD.py, Fiduciallnfo.py, 
ObjectInfo.py, blocks.py, projection_helper.py.

## [1.3.36] - 2023-05-15
### Changed
- Changed geotiff2deg.py, user must now explicitly specify DEM reference
surface, or accept default value of EGM2008.
- Changed converter.py, and convert_to_sicd.py updated to reflect changes 
to geotiff2deg.py.

## [1.3.35] - 2023-05-08
### Fixed
- Fixed bug with readthedocs correctly building documentation per 
[readthedocs.org issue #10290](https://github.com/readthedocs/readthedocs.org/issues/10290)
and [sphinx-doc issue #10378](https://github.com/sphinx-doc/sphinx/issues/10378).

## [1.3.34] - 2023-05-08
### Added
- Added X1Y1 vs X2Y2 checks to cphd_consistency and test_cphd_consistency.

## [1.3.33] - 2023-05-02
### Added
- Added support for DEMs in GeoTIFF format.
- Added projection to a DEM for SCP and image corner coordinates in 
sarpy/utils/convert_to_sicd.py and sarpy/io/complex/converter.py modules 
when a GeoTIFF DEM file is used as a CLI argument in convert_to_sicd program. 

## [1.3.32] - 2023-05-01
### Fixed
- Fixed bug in SARPY correctly reading numerous nitf files in linux OS.
### Changed
- Minor cleaning in nitf.py.

## [1.3.31] - 2023-04-28
### Fixed
- Fixed bug in SARPY correctly reading data with odd number of rows with a 
stride that is not a multiple of the input size.

## [1.3.30] - 2023-04-24
### Added
- Added OSR checks to consistency.
### Fixed
- Fixed misspelling of DESOFLW in des.py

## [1.3.29] - 2023-04-24
### Added
- Added Radiometric and Timeline.IPP nodes to tests/data/example.sicd.xml
- Added Radiometric and Timeline.IPP nodes to tests/data/example.sicd.rma.xml
- Added tests for Radiometric.py, RgAzComp.py, Timeline.py, ImageData.py, 
and RadarCollection.py
to tests/io/complex/sicd_elements/test_sicd_elements.py

### Changed
- Updated code to align with PEP.

### Fixed
- Fixed bug in Sarpy correctly raising error when testing length of array in 
- sarpy/io/complex/sicd_elements/ImageData.py FullImageType
- Fixed bug in Sarpy correctly returning frequency band name for Hz in 
sarpy/io/complex/sicd_elements/RadarCollection.py
- Fixed bug in Sarpy correctly raising error when testing length of array in 
RadarCollection.py TxFrequencyType.from_array.
- Fixed bug in Sarpy correctly setting RcvFMRate in RadarCollection.py WaveformParametersType.

## [1.3.28] - 2023-04-21
### Fixed
- Fixed bug in Sarpy correctly handling negative longitudes when calculating 
longitude grid size in point_projection.image_to_ground_dem.

## [1.3.27] - 2023-04-17
### Added
- Tests for utils, validation_checks, PFA, RMA, and Grid
### Fixed
- Fixed bug in Sarpy correctly calling _derive_unit_vector_params with correct RMAParam.
- Fixed bug in Sarpy correctly checking Grid.type is RGAZIM and resetting type if not.
- Fixed bug in Sarpy returning 0 and not None in _derive_rg_az_comp 
when Grid.Col.DeltaKCOAPoly does not exist.

## [1.3.26] - 2023-03-17
### Changed
- Moved test_consistency and test_cphd_consistency from pytests folder to tests folder. 
### Fixed
- Fixed bug in SarPy correctly reading SIDDs with undeclared XML.
- Fixed bug in CPHD correctly using AmpScaling when reading CPHD signal arrays with.
integer types that have an AmpSF PVP.
- Fixed bug in CPHD consistency by updating tolerances and use of Approx in numpy comparisons.
### Added
- Added test for SIDD undeclared XML namespace.
- Added additional checks for cphd_consistency dwell polynomials.

## [1.3.25] - 2023-03-10
### Added
- Added additional checks for micro-parameters, and related metadata in cphd_consistency.
### Changed
- Improved cphd_consistency result formatting.
### Fixed
- Fixed GainPhaseArrayType._numeric_format typo when creating the dictionary. 
 
## [1.3.24] - 2023-03-08
### Added
- Added more coverage to test_geometry_elements

## [1.3.23] - 2023-02-23
### Fixed
- Fixed pixel_type check for get_format_function within SICDWriter.
### Added
- Added additional information in documents for writer class close() function 
to explicitly state that not closing file after writing can raise ValueErrors
when trying to read them. 

## [1.3.22] - 2023-02-23
### Added
- Added point projection tests.
 
## [1.3.21] - 2023-02-22
### Fixed
- Fixed bug with applying the correct endianness being applied when reading SIO data. 

## [1.3.20] - 2023-02-16
### Fixed
- Fixed bug with walrus operator and other modern syntax to ensure code is backwards 
compatible with older python versions. 3.8 and below.
### Changed
- Changed CPHD Consistency to remove antenna file tests. 

## [1.3.19] - 2023-02-13
### Fixed
- Fixed bug in check_channel_dwell_exist message
### Added
- Added auto-detection of schema for CPHD data.
- Added new check_file_type header check in tests.
- Added new check_identifier_uniqueness check in tests.
- Added new check_channel_normal_signal_pvp check in tests.
- Added new TOAExtended.TOAExtSaved check in tests.
- Added two new optional PVP checks, check_optional_pvps_fx, and check_optional_pvps_toa
- Added passes and skips to ConsistencyChecker
### Changed
- Changed cphd_consistency testing dependencies to be optional, skips the check if networkx, 
and shapely are not installed.
- Simplified pytest.skip usage in test_cphd_consistency.

## [1.3.18] - 2023-02-10
### Fixed
- Fixed a bug in SICD get_format_function correctly checking for PixelType.
### Added
- Added more tests to test_geocoords.py, switched from unittests to pytest.

## [1.3.17] - 2023-02-06
### Fixed
- Fixed bug in SICD converter returning 1D array instead of a 2D array.
### Added
- Added close connection for test_remote. 

## [1.3.16] - 2023-02-02
### Changed
- Misleading naming/comment for DMS
### Added
- Added Tests for latlon.

## [1.3.15] - 2023-02-02
### Fixed
- Fixed bug in intersection and bounding box code for geometry elements.
### Added
- Added Tests for geometry.

## [1.3.14] - 2023-01-11
### Fixed
- Fixed deprecated usage of numpy.bool
### Changed
- Updated documentation on usage of CPHD consistency check.

## [1.3.13] - 2022-11-21
### Fixed
- Fixed bug in correctly setting attribution for populate_nitf_information_into_sicd().
### Changed
- Changed links for SAR standards in the readme file, pointed links to the newest versions.

## [1.3.12] - 2022-11-15
### Fixed
- Fixed accuracy of TimeCOA and DeltaKCOA polygons for ICEYE Spotlight.

## [1.3.11] - 2022-11-08
### Fixed
- Fixed bug with conditional testing of NITF image header for jpeg2k files correctly 
testing and applying format for C8/regular and M8/with masking. 
### Added
- Added links to SAR data for Capella, NISAR, and CPHD file formats.
### Changed
- Changed links for general NITF, Radarsat-2, and PALSAR files.

## [1.3.10] - 2022-10-24
### Fixed
- Updated error message for IQ handing in CPHD reader.
- Refactored reference to MIL-STD-2500C NITF spec to "Joint BIIF Profile (JBP)".
- Fixed bug in correctly accounting for row stride from file like object.
[Pull Request 348](https://github.com/ngageoint/sarpy/pull/348)
- Fixed bug with SICD converter correctly conform to described relationship in the SICD D&I.
[Pull Request 349](https://github.com/ngageoint/sarpy/pull/349)
- Fixed bug with Grid.Col.DeltaKCOA poly correctly populated based on collection metadate 
and constant COA. [Pull Request 350](https://github.com/ngageoint/sarpy/pull/350)
### Added
- Added remap.py unit test. [Pull Request 351](https://github.com/ngageoint/sarpy/pull/351)

## [1.3.9] - 2022-10-13
### Fixed
- Fixed a bug in correctly handling IQ error in CPHD reader.
- Refactored CHPD schema file name to match official release. 

## [1.3.8] - 2022-10-07
### Fixed
Fixed a bug in correctly cleaning up during closing of nitf reader.

## [1.3.7] - 2022-08-29
### Fixed
Fixed a bug in correctly handling COSAR version 2 files, for TerraSAR-X.

## [1.3.6] - 2022-08-23
### Fixed
Fixed a bug in improper raw writing for a CPHD/CRSD file.

## [1.3.5] - 2022-08-11
### Fixed
Fixed a bug in improper checking of whether the signal data for a CPHD/CRSD file 
was fully written.

## [1.3.4] - 2022-07-01
### Changed
- Added SICD version 1.3.0 information, and set the default SICD version number 
accordingly.
- Added CPHD version 1.1.0 changes, and set the default CPHD version number 
accordingly.
- Set the stage for anticipated CRSD changes.
- Removed the colon character with an underscore character in a suggested file 
name, since apparently colon in not permitted in a Windows file name.
### Fixed
Using NamedTemporaryFile in unit tests on writing files fails on Windows, and 
usage has been replaced.

## [1.3.3] - 2022-06-20
### Fixed
Resolved bug in TerraSAR-X window weighting determination.

## [1.3.2] - 2022-06-09
### Fixed
Resolved bug in data segment for reading from remote data source.

## [1.3.1] - 2022-06-08
### Fixed
Resolved type-hinting bug for missing optional h5py dependency.

## [1.3.0] - 2022-06-06
### Changed
- The base reading and writing structures have been updated to enable reading and 
writing data in both the natural use format of data (using `read()` or `write()`), 
as well as the raw storage format (using `read_raw()` or `write_raw()`). 
- kmz construction has been moved from the `sarpy.io.product` subpackage to the 
`sarpy.visualization` subpackage.
- The `sarpy.processing` subpackage has been restructured for clarity of purpose.
This includes moving sidd production construction has been moved from the 
`sarpy.io.product` subpackage to the `sarpy.processing.sidd` subpackage.

### Added
Implementations for DataSegment and FormatFunction for reading and writing 
changes.

# 1.3


## [1.2.70] - 2022-05-05
### Changed
Helper functions for determine radar band name have been changed to return the 
band name of the center frequency versus an aggregate of the band name of the 
maximum and minimum frequencies.

## [1.2.69] - 2022-04-18
### Added
A preliminary version of the "regi" registration method implementation

## [1.2.68] - 2022-04-14
### Fixed
CPHD Version 1.0 parsing of GeoInfo Polygon and Line element has been fixed

## [1.2.67] - 2022-04-12
### Added
Added support for reading SICD version 1.3, full integration will not be complete 
until the format is fully finalized and approved

## [1.2.66] - 2022-04-06
### Fixed
Improved construction of LinearRing geojson element

## [1.2.65] - 2022-03-25
### Fixed
- Resolved minor bug involved with xml namespace handling
- Corrected layover calculation in AFRL object population

## [1.2.64] - 2022-03-22
### Fixed
Fixed a bug associated with a plane projection where the column unit vector was 
misidentified

## [1.2.63] - 2022-03-16
### Added
Created a few methods for parsing a sicd structure directly from a xml file

## [1.2.62] - 2022-03-08
### Changed
- Made md5 checksum calculation optional for afrl/rde construction
- Permit an optional call which returns of raw values (not cast to unit8/16) 
  from remap functions

### Fixed
Fixed the PEDF remap calculation

## [1.2.61] - 2022-03-02
### Fixed
Resolving bug/error in NITF segmentation scheme and the population of SICD/SIDD 
segmentation details

## [1.2.60] - 2022-03-01
### Fixed
Resolving BIP chipper definition/application bug for segmented files

## [1.2.59] - 2022-02-23
### Fixed
Update for CSG metadata `Doppler Rate vs Azimuth Time Polynomial` location change

## [1.2.58] - 2022-02-17
### Fixed
Resolved bug in NITF with compression, but the support is still only partial and 
fairly fragile

## [1.2.57] - 2022-02-15
### Changed
Incorporating changes to AFRL/RDE labeling structure agreed upon on 2022-02-15

## [1.2.56] - 2022-02-07
### Fixed
Fixed a bug in scaling failure for the creation of KMZ from SICD

### Added
- A ReferencePoint property to SIDD.Measurement
- Basic image registration functions for 
    - finding the best adjustable projection parameters to fit known geolocation and 
    observed image location
    - find the best geophysical location given a collection of images and observed 
    image locations

## [1.2.55] - 2022-02-01
### Changed
Modified the RGIQE parameters to a more agreed upon version

## [1.2.54] - 2022-01-31
### Changed
Finalizing changes to the AFRL/RDE structure agreed upon on 01/26/2022

## [1.2.53] - 2022-01-25
### Changed
Add a CPHD validation check for existence of ImageGrid

### Fixed
Resolves a potentially invalid gc.collect call

## [1.2.52] - 2022-01-19
### Added
- Add links to NGA standards documents in the readme
- Introduction of a subset SICD reader class implementation

### Changed
Update to AFRL/RDE based on agreed upon changes

### Fixed
- Documentation fixes
- Resolves projection to DEM bug in expected shape of results

## [1.2.50] - 2022-01-11
### Fixed
- Bug fixes for AFRL label structure

## [1.2.49] - 2022-01-07
### Fixed
Resolving some error catching when trying to open vanilla tiff

## [1.2.48] - 2022-01-05
### Fixed
Resolves bug which resulted in empty columns in orthorectified image

## [1.2.47] - 2021-12-30
### Fixed
Corrects a number of bugs and code smells revealed by SonarQube

## [1.2.46] - 2021-12-29
### Added
- Introduction of rational polynomial fitting tools
- New projection helper using rational polynomial model (for speed)

### Changed
- Split original AFRL label structures and AFRL/RDE structures for NGA specific 
usage
- Split sarpy.processing.orthorectification module into subpackage

## [1.2.45] - 2021-12-20
### Fixed
Correction of population of SICD ModeType for specific ICEYE collection mode SLC

## [1.2.44] - 2021-12-06
### Fixed
Correction of minor bugs in AFRL label structure population and serialization

## [1.2.43] - 2021-11-30
### Changed
More improvements to Capella SLC reader. *(Remains an incomplete work-in-progress.)*
 
### Fixed
- Completes resolution of bug with unicode handling in sicd validation from 1.2.41
- Correction to SICD and SIDD writing for image segmentation to comply with 
standards requirements. *(Before here, "large" images with more than 99,999 
columns were incorrectly segmented.)*

## [1.2.41] - 2021-11-16
### Fixed
Resolves bug with unicode handling in sicd validation. 
*(This proves to be incomplete.)*

## [1.2.40] - 2021-11-15
### Changed
Updates AFRL label structures with agreed changes

## [1.2.39] - 2021-11-04
### Fixed
- Fixes (edge case) computation of doppler centroid polynomials for Cosmo SkyMed 
2nd generation SLC reader
- Fixes logging configuration issues for command-line utilities

## [1.2.38] - 2021-11-03
### Changed
- Some improvements to the Capella SLC reader, but support remains incomplete
- Adds a check for SICDTypeReaders to ensure that chipper sizes agree with 
expected values

## [1.2.37] - 2021-11-02
### Added
Adds a `CollectEnd` property for sicd.Timeline of type `numpy.datetime64`

### Changed
Centralizing the xml floating point serialization format. Reverts to format `G` 
for many fields, in relation to changes form 1.2.34.

## [1.2.36] - 2021-10-27
### Changed
- Refactor of the sarpy.annotation structures and elements for improved clarity.
- Extensions, bug fixes, and improvements of geometry object definitions.

## [1.2.35] - 2021-10-25
### Changed
Changes the Radiometric polynomials for SICD subaperture degradation and the 
rgiqe methods for degrading to a desired quality. *(This proves to be incorrect.)*

## [1.2.34] - 2021-10-22
### Changed
Serialize most xml fields of type double using 17 decimals, versus 16, for full 
precision, and uses the exponential format for most. 
*(Modified in the immediate future.)*

## [1.2.32] - 2021-10-21
### Fixed
Bug fix for left looking radarsat SLC data

## [1.2.31] - 2021-10-18
### Added
Adds a command-line utility for populating a SICD nominal noise polynomial, 
given the presence of other RCS polynomials

### Fixed
Account for noise and Radiometric SF poly changes due to subaperture and 
weighting changes in sarpy.processing.sicd.normalize_sicd methods

## [1.2.29] - 2021-10-15
### Fixed
- Corrects DeltaKCOA and DopCentroidPoly population for Cosmo SkyMed 2nd generation
- Bug fix SICD.RadarCollection extra parameter parsing

## [1.2.28] - 2021-10-13
### Fixed
- Bug fixes for sicd validation of version 1.1 date and pfa bounds consistency 
checking for the spotlight case
- Ensure that FTITLE and IID2 nitf header fields in a sicd file will be prefixed 
with 'SICD:' if attempting to write a version 1.1 SICD file
- Correcting data for SICD version 1.1.0 schema

## [1.2.27] - 2021-10-09
### Fixed
Bug fix for left facing TerraSAR-X SLC data symmetry orientation

## [1.2.26] - 2021-10-04
### Added 
Adds CPHDTypeReader and CRSDTypeReader abstract class definitions for better 
inheritance structure.

### Changed
- Moves CRSD implementation from sarpy.io.phase_history to new subpackage 
sarpy.io.received for conceptual clarity.
- Replaces the bad slice convention parsing from the __call__ method with
        the standard expected for Python

## [1.2.24] - 2021-10-01
### Added 
- Adds and AbstractReader class definition for better inheritance structure. 
- Adds SICDTypeReader and SIDDTypeReader abstract class definitions for better 
inheritance structure.

### Changed
Remap methods changed from flat functions to callable class implementation, for 
more clear preservation of remap function state variables

### Fixed
Bug fix for file reader `fileno` property usage

## [1.2.23] - 2021-09-29
### Added
Introduces RGIQE methods with sicd degradation functions and a method for 
producing a sicd/reader with reweighting and/or subaperture processing applied

### Changed
- Adjustments of unit tests to skip versus fail missing tests for CPD validation. 
- Refactors some elements of SICD window population to use the 
sarpy.processing.sicd.windows module.

### Removed
Drops stated support for Python 2.7

## [1.2.22] - 2021-09-20
### Changed
Includes population of North in SIDD exploitation features construction

### Fixed
Fixes deprecated numpy.object type usage

## [1.2.21] - 2021-09-17
### Fixed
- Bug fix for default sicd conversion naming scheme.
- Bug fix for Capella SLC symmetry orientation.

## [1.2.19] - 2021-09-16
### Added
Creates command-line utility for SICD chip production

### Changed
Unified NITF header field value preservation in sicd and sidd reader, writer, 
and conversion 

## [1.2.18] - 2021-09-15
### Changed
Creates a few methods in the sicd reader to permit preservation of more nitf 
fields when copying a sicd

## [1.2.17] - 2021-09-13
### Changed
Updates the preliminary Capella SLC format reader to accommodate SPOTLIGHT mode 
data. *This remains a work-in-progress.*

## [1.2.16] - 2021-09-10
### Changed
Updates the preliminary Capella SLC format reader to the structure of the actual 
data, and dropping support for the deprecated pre-release format data. *This 
remains a work-in-progress.*

## [1.2.15] - 2021-09-08
### Added
Creates sarpy.processing.sicd.windows module for more unified handling of commonly used
Fourier windowing definitions and functions

### Fixed
Tidies up deprecated numpy.bool definitions

## [1.2.14] - 2021-09-02
### Added
Adds support for ENU coordinate system data in geocoords transform paradigm

## [1.2.13] - 2021-08-31
### Added
- Adds upport for reading GFF version 1.6, 1.8, and 2.X files.
- Creates some helper functions/classes for AFRL label data construction.

## [1.2.12] - 2021-08-12
### Changed
xml handling moved to a centralized sub-package for conceptual consistency 
and clarity

### Added
Initial effort for enabling interpretation and production of labelled image data
following the AFRL labeling scheme

### Fixed
Attempt at CPHD validation fix for PVP values

## [1.2.11] - 2021-08-04
### Changed
Moves to module based logging scheme, which permits sarpy specific log 
configuration

## [1.2.10] - 2021-07-22
### Fixed
Minor bug fixes for CPHD fields

### Added
Adds support for reading and writing Compensated Received Signal Data (CRSD) 
format files and data

## [1.2.9] - 2021-07-16
### Fixed
Fixes error in Cosmo SkyMed SLC mode handling

## [1.2.8] - 2021-07-14
### Fixed
Corrects population of SICD metadata 2nd generation Cosmo SkyMed SLC QuadPol 
collections correctly

## [1.2.7] - 2021-06-30
### Changed
More SICD validation corrections and improvements

## [1.2.6] - 2021-06-29
### Fixed
Fixes a bug in the create_kmx command line usage

### Changed
- Improves sicd validation, including checks for values in GeoData.ImageCorners, 
GeoData.ValidData, and ImageData.ValidData, check consistency between 
RadarCollection.RcvChannels and ImageFormation.RcvChanProc, check consistency 
between sicd structure and the appropriate NITF DES subheader values and Image 
segment structure.
- Improves sidd validation by checking consistency between sidd structure and the 
appropriate NITF DES subheader values and Image segment structure.

## [1.2.5] - 2021-06-21
### Fixed
Correct check if SICD Grid.Type is one of expected types

## [1.2.4] - 2021-06-16
### Added
Replaces the use of IOError with custom error, which simplifies and improves 
error handling

## [1.2.2] - 2021-06-14
### Fixed
Bug fixes for NITF 2.0 implementation and NITF LUT usage

## [1.2.1] - 2021-06-11
### Fixed
Bug fixes for complex NITF construction, the Poly2DType shift method, the 
BSQChipper, and NITF writing for an extra long/wide image

## [1.2.0] - 2021-05-27
### Changed
Initialized with streamlining and simplification of reader structure

# 1.2

## [1.1.78] - 2021-05-24
Improving documentation for geometry/projection elements

## [1.1.77] - 2021-05-20
Bug fix in writing MONO16I SIDD

## [1.1.76] - 2021-05-18
Refining examples and incorporating documentation hosting at readthedocs

## [1.1.75] - 2021-05-12
Correcting bugs in the nrl, linear, and log remap functions

## [1.1.74] - 2021-05-11
Introducing the ability to read SICD, SIDD, and some NITF files from file-like
object. This is generally geared towards usage with smart_open for possible S3 usage.

## [1.1.73] - 2021-04-30
- Corrected another mistake in the SIDD version 2 schema and element production
- Corrected some mistakes in the included SIDD version 2 schema

## [1.1.71] - 2021-04-29
Introducing basic SIDD consistency check, and debugging the squint calculation

## [1.1.70] - 2021-04-28
- Debugging SIDD Version 2.0 structure produced in create_product methods
- Debugging of poor classification extraction in SIDD production for both versions
- Reorganization of SIDD schemas and inclusion in package data

## [1.1.69] - 2021-04-27
- Debugging SIDD Version 1.0 structure produced in create_product methods
- Introduction of a reader implementation which directly uses an array or memmap,
which is intended merely to provide unified integration for tool usage.

## [1.1.67] - 2021-04-19
Fix for the range doppler rate polynomial definition for Cosmo SkyMed

## [1.1.66] - 2021-04-15
- Updating Cosmo SkyMed 2nd generation column impulse response population
- Introduces helper method for converting image coordinates to coordinates for
polynomial evaluation
- Fixed sign convention in SICD related Fourier helper methods
- Fixed state issues with ortho-rectification model populated from
sicd.RadarCollection.Area

## [1.1.65] - 2021-04-08
CCINFA tre definition bug fix and SICD AmpTable usage bug fix

## [1.1.64] - 2021-04-05
- Loosening the handling for poorly formed SICD structure
- Making file type discovery logging less verbose

## [1.1.62] - 2021-03-30
CPHD Consistency Improvements

## [1.1.61] - 2021-03-24
Introduction of Cosmo SkyMed 2nd generation support

## [1.1.60] - 2021-03-23
Creating a command line utility to dump some CPHD metadata

## [1.1.59] - 2021-03-22
- Implementing support for reading masked NITF without compression, read band
sequential format, and correction of band interleaved by block reading.
- What remains is more general support for compression, which is likely not high
on the list of priorities.

## [1.1.58] - 2021-03-15
- Creating command line utility for creating SIDD products from a SICD type reader
- Creating command line utility for creating KMZ products from a SICD type reader
- Streamlining the command line utility for performing a dump of a NITF header

## [1.1.57] - 2021-03-09
- Fixing tre loop parsing issue for ACCHZB, OBJCTA, and CCINFA
- Final verification for SICD.RadarCollect.Area for definition for default SIDD image bounds

## [1.1.55] - 2021-03-08
Bug fixes for CPHD writing

## [1.1.54] - 2021-03-05
Further debugging SICD.RadarCollect.Area for definition for default SIDD image bounds

## [1.1.53] - 2021-03-04
Introducing CPHD 1.0 writing capabilities and improved CPHD reading for both versions

## [1.1.52] - 2021-03-02
Use SICD.RadarCollect.Area for definition for default SIDD image bounds

## [1.1.51] - 2021-02-16
Introduced validity check for SICD in the consistency module

## [1.1.50] - 2021-02-15
Imposing print function compliance for Python 2.7 usage

## [1.1.49] - 2021-02-12
Completion of annotation and geometry elements for apps usage

## [1.1.48] - 2021-02-05
- Bug fixes for CPHD AntGainPhase support array reading
- Labeling schema modifications and geometry elements modifications

## [1.1.47] - 2021-02-02
- Modification of extracted CSK metadata for proper ImpRespBW population
- Top level structures for SICD, SIDD, and CPHD better handle default xml tag
- Change to aperture_filter for sarpy_apps usage
- Changes and fixes in geometry_elements for sarpy_apps usage

## [1.1.46] - 2021-01-25
Valkyrie added CPHD consistency checked and some CPHD associated unit tests

## [1.1.45] - 2021-01-05
- Refining some validation parameters checks
- Refining CMETAA usage, annotation schema improvements, and nitf header methods

## [1.1.43] - 2020-12-21
Clean-up of opener definitions and functionality

## [1.1.42] - 2020-12-18
- Minor fix to setup.py definition
- Added reader_type property for reader to clarify intent and usage

## [1.1.40] - 2020-12-17
Debugging CMETAA definition and usage

## [1.1.39] - 2020-12-16
Debugging the Exploitation Calculator for SIDD population

## [1.1.38] - 2020-12-15
Adjust TRE definitions to avoid errors in improperly parsing numeric fields

## [1.1.37] - 2020-12-13
Adding preliminary support for NITF 2.0

## [1.1.36] - 2020-12-10
Minor adjustment for some basic NITF header elements for bug fixes

## [1.1.35] - 2020-12-08
Removal of deprecated code and revamp/expansion of unit testing

## [1.1.34] - 2020-12-02
Correction for difference between KompSat and CSK metadata

## [1.1.33] - 2020-12-01
- Correction of radar mode determination for KompSat
- Bug fixes for Cosmo Skymed and fine-tuning KompSat support

## [1.1.31] - 2020-11-25
Bug fixes for sicd structure serialization

## [1.1.30] - 2020-11-20
- Make the SICD version 1.1 creation an option, versus the default
- Created SICD file will now be version 1.1, if the polarization permits

## [1.1.28] - 2020-11-13
Fixed bug in PFA validation inspection

## [1.1.27] - 2020-11-06
Minor debugging of CSK metadata

## [1.1.26] - 2020-11-04
- Adding more robust SICD validation tests
- README update for installation

## [1.1.24] - 2020-10-21
Debugging PALSAR ADC rate population and improving some docstrings

## [1.1.23] - 2020-10-19
Added PALSAR (ALOS2) reading support for level 1.1 products

## [1.1.22] - 2020-10-13
Making suggested name more resistant to errors

## [1.1.21] - 2020-10-08
Adding RCM NITF format data support

## [1.1.20] - 2020-10-07
- SICD naming scheme separation
- Added TerraSAR-X reading support for level 1 products
- Repaired ICEYE/CSK data reading bug which omitted last row

## [1.1.17] - 2020-10-06
Repaired ICEYE and CSK chipper definition bug

## [1.1.16] - 2020-09-25
Treat RCM ScanSAR as spotlight for appropriate processing

## [1.1.15] - 2020-09-24
Resolved SICD writing bug for integer data

## [1.1.14] - 2020-09-23
Improved docstrings on RCM ScanSAR methods

## [1.1.13] - 2020-09-21
Bug fix for RSMGGA tre element

## [1.1.12] - 2020-09-18
- Added ScanSAR support for RCM reader
- Bug fixes for some NITF header element parsing
- Very basic example refinement

## [1.1.9] - 2020-09-15
Introducing more general complex valued NITF support and more general BIP support

## [1.1.8] - 2020-09-08
Fixing polygon inclusion bug for orientation

## [1.1.7] - 2020-09-03
Bumping the required numpy version so datetimes are handled correctly

## [1.1.6] - 2020-09-02
Bug fixes for ICEYE data ordering, RCM classification string, and DTED bounding box definition

## [1.1.5] - 2020-09-01
- Improvements and debugging for DTED and projection to DEM
- Changed some functionality in aperture filter

## [1.1.3] - 2020-08-25
- Added helper methods for aperture tool processing
- Fixed bug for NITF block ordering
- Improvements to change detection support

## [1.1.0] - 2020-08-21
Initialized with more general NITF reading support, including supporting compressed
image segments, and introduced preliminary capability for reading a standard
change detection package

# 1.1

## [1.0.53] - 2020-08-
SICD normalization method bug fixes

## [1.0.52] - 2020-08-13
- Fine-tuning parameter defaults and naming in projection methods
- Fixed subtle bug in projection to constant HAE method
- Introduced projection methods for SIDD structures
- Bug fix for SICD normalization methods

## [1.0.48] - 2020-08-12
Making SICD normalization methods more robust

## [1.0.47] - 2020-08-10
- Tidy up ICEYE parameter values
- Tidy up chipper argument values

## [1.0.45] - 2020-08-07
- Introduced ICEYE reading capability
- Handling some edge cases for ortho-rectification issues

## [1.0.43] - 2020-08-06
Fixed bug in SICD.RadarCollection.Area and SICD.MatchInfo parsing

## [1.0.42] - 2020-08-05
- Introduced coherent KMZ and SIDD product creation
- Resolved CPHD Version 1.0 bug for PVP parsing

## [1.0.40] - 2020-07-29
- Adjust ImageBeamComp for derived formats
- Account for non-convexity of projection for ortho-rectification

## [1.0.38] - 2020-07-28
- Clarified some projection parameters and documentation
- Optimized some ortho-rectification parameters

## [1.0.36] - 2020-07-27
Refactored CSI and subaperture processing and introduced practical SIDD creation

## [1.0.35] - 2020-07-21
Added support for Capella format data

## [1.0.34] - 2020-07-15
Fixed NITF bug for image size

## [1.0.33] - 2020-07-14
Some general NITF compatibility updates

## [1.0.32] - 2020-07-13
Added support for blocked NITF files and improved NITF header handling

## [1.0.31] - 2020-07-10
Correct minor issues for CPHD and xml interpretation

## [1.0.30] - 2020-07-09
Provide aggregate complex type reader, and sicd partitioning methods

## [1.0.29] - 2020-07-08
- Various fixes for tiff reading and reorganization
- Reorganization of sarpy.io structure into proper categories

## [1.0.27] - 2020-07-02
- General purpose ortho-rectification capabilities introduced.
- 2020-06-Introduce file_name property for reader objects

## [1.0.25] - 2020-06-30
Resolving usage of numpy.stack to properly allow numpy 1.9.

## [1.0.24]
Bug fix for the CPHD version 0.3 GlobalType definition.

## [1.0.23]
Moved the general tk_builder, and specific sarpy_apps into their own repos

## [1.0.22]
Cleaned up error production and catching for file opening effort

## [1.0.21]
Introduced basic CPHD (0.3 and 1.0) reading capability

## [1.0.20]
Introduced NISAR reading capability

## [1.0.19]
Base SIDD reading and writing capability

## [1.0.18]
Bug fix for appropriate error catching during file reading effort

## [1.0.17]
Bug fix for point projection method

## [1.0.16]
Fixing BigTiff error

## [1.0.15]
Introducing suggested name capability for the SICD

## [1.0.14]
Making security tag setting in NITF more flexible, and fixing bug

## [1.0.13]
Introduction of metaicon gui element

## [1.0.12]
Introduction of basic SIDD elements, still experimental.

## [1.0.11]
Debugging RNIIRS population

## [1.0.10]
Defining a few more relevant SICD.SCPCOA properties

## [1.0.9]
Moving latlon functions back into main branch

## [1.0.8]
Bug fix for polarization data and population of RNIIRS

## [1.0.7]
Small bug fix in converter file naming scheme process

## [1.0.6]
Fixing a small but fatal bug for TRE parsing

## [1.0.5]
Modified string enum parsing for NITF header to make data errors non-fatal

## [1.0.4]
Improvements for GUI behavior and annotation tool

## [1.0.3]
Refactored NITF elements for clarity and better documentation

## [1.0.2]
Made polynomial shift simpler and added for Poly2D

## [1.0.1]
Added a very basic ccd calculation and two SICD properties

## [1.0.0]
Start of version after major refactor
