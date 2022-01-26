# Change Log

SarPy follows a continuous release process, so there are lots of releases. Since 
essentially every commit corresponds to a release, specific release points are 
not being annotated in github.

## [Unreleased]

Words

## [1.2.54] - 2022-01-
### Added
Preliminary registration methods based on the Sandia `regi` methodology


## [1.2.53] - 2022-01-25
### Changed
Add a CPHD validation check for existence of ImageGrid

### Fixed
Resolves a potentially invalid gc.collect call

## [1.2.52] - 2022-01-19
### Added
- Adds links to NGA standards documents in the readme
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
weighting changes in sarpy.processing.normalize_sicd methods

## [1.2.29] - 2021-10-15
### Fixed
- Corrects DeltaKCOA and DopCentroidPoly population for Cosmo SkyMed 2nd generation
- Bug fix SICD.RadarCollection extra parameter parsing

## [1.2.28] - 2021-10-13
### Fixed
- Bug fixes for sicd validation of version 1.1 date and pfa bounds consistency 
checking for the spotlight case
- Ensure that FTITLE and IID2 nitf header fields in a sicd file will be prefixed 
with 'SICD:' if attepting to write a version 1.1 SICD file
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
sarpy.processing.windows module.

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
Creates sarpy.processing.windows module for more unified handling of commonly used
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

## Earlier version omitted...