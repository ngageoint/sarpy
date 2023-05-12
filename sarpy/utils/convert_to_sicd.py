"""
Convert from complex SAR image format to SICD format.

For a basic help on the command-line, check

>>> python -m sarpy.utils.convert_to_sicd --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Valkyrie Systems Corporation")

import argparse
import logging

import sarpy
from sarpy.io.complex.converter import conversion_utility


def convert(input_file, output_dir, preserve_nitf_information=False,
            dem_filename_pattern=None, dem_type=None, geoid_file=None):
    """

    Parameters
    ----------
    input_file : str
        Path to the input file.
    output_dir : str
        Output directory path.
    preserve_nitf_information : bool
        Try to preserve NITF information? This only applies in the case that the
        file being read is actually a NITF file.
    dem_filename_pattern : str | None
        Optional string specifying a Digital Elevation Model (DEM) filename pattern.
        This is a format string that specifies a glob pattern that will
        uniquely specify a DEM file from the Lat/Lon of the SW corner of
        the DEM tile.  See the convert_to_sicd help text for more details.
    dem_type : str | None
        Optional DEM type ('GeoTIFF', 'GeoTIFF:WGS84', 'GeoTIFF:EGM2008', etc.).
        This parameter is required when dem_filename_pattern is specified.  For 'GeoTIFF'
        DEM files, the reference surface can be either WGS84 or any of the geoid models.
        The reference surface is appended to the DEM type with a ':' separator.  If the
        reference surface is not specified, then EGM2008 is assumed.
    geoid_file : str | None
        Optional Geoid file which might be needed when dem_filename_pattern is specified.
    """

    conversion_utility(input_file, output_dir, preserve_nitf_information=preserve_nitf_information,
                       dem_filename_pattern=dem_filename_pattern, dem_type=dem_type, geoid_file=geoid_file)


if __name__ == '__main__':
    epilog = ('Note:\n'
              'The DEM files must have the SW corner Lat/Lon encoded in their filenames.\n'
              'The --dem-path-pattern argument contains a format string that when populated will\n'
              'create as glob pattern that will specify the desired DEM file.  The following\n'
              'arguments are provided to the format string.\n'
              '    lat = int(numpy.floor(lat))\n'
              '    lon = int(numpy.floor(lon))\n'
              '    abslat = int(abs(numpy.floor(lat)))\n'
              '    abslon = int(abs(numpy.floor(lon)))\n'
              '    ns = "s" if lat < 0 else "n"\n'
              '    NS = "S" if lat < 0 else "N"\n'
              '    ew = "w" if lon < 0 else "e"\n'
              '    EW = "W" if lon < 0 else "E"\n'
              '\n'
              'For example (with Linux file separators) the following specifies a GeoTIFF DEM pattern:\n'
              '    /dem_root/tdt_{ns}{abslat:02}{ew}{abslon:03}_*/DEM/TDT_{NS}{abslat:02}{EW}{abslon:03}_*_DEM.tif\n'
              '\n'
              'In theory, one could use a simple format string using wildcard characters like this:\n'
              '    /dem_root/**/*{NS}{abslat:02}{EW}{abslon:03}*DEM.tif\n'
              '\n'
              'However, this would be unwise since glob might have to scan through many files and directories\n'
              'to find the desired file.  This could be quite time consuming if there are many files in dem_root.\n'
              )

    parser = argparse.ArgumentParser(description="Convert to SICD format.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=epilog)
    parser.add_argument(
        'input_file', metavar='input_file',
        help='Path input data file, or directory for radarsat, RCM, sentinel or other systems.\n'
             '* For radarsat or RCM, this can be the product.xml file, or parent directory\n'
             '  of product.xml or metadata/product.xml.\n'
             '* For sentinel, this can be the manifest.safe file, or parent directory of\n'
             '  manifest.safe.\n')
    parser.add_argument(
        'output_directory', metavar='output_directory',
        help='Path to the output directory. This directory MUST exist.\n'
             '* Depending on the input details, multiple SICD files may be produced.\n'
             '* The name for the output file(s) will be chosen based on CoreName and\n '
             '  transmit/collect polarization.\n')
    parser.add_argument(
        '-p', '--preserve', action='store_true',
        help='Try to preserve any NITF information?\n'
             'This only applies in the event that the file being read is a NITF')
    parser.add_argument(
        '-d', '--dem-filename-pattern',
        help='Optional string specifying a Digital Elevation Model (DEM) filename pattern.\n'
             'This is a format string that specifies a glob pattern that will\n'
             'uniquely specify a DEM file from the Lat/Lon of the SW corner of\n'
             'the DEM tile.  See the note below for more details.\n')
    parser.add_argument(
        '-t', '--dem-type',
        help=('Optional DEM type ("GeoTIFF", etc.).\n'
              'This parameter is required when dem-path-pattern is specified.\n'))
    parser.add_argument(
        '-g', '--geoid-file',
        help='Optional path to a geoid definition file.\n'
             'A geoid definition file is required when dem-path-pattern is specified\n'
             'and the DEM height values are relative to a geoid.\n')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose (level="INFO") logging?')

    args = parser.parse_args()
    level = 'INFO' if args.verbose else 'WARNING'
    logging.basicConfig(level=level)
    logger = logging.getLogger('sarpy')
    logger.setLevel(level)

    convert(args.input_file, args.output_directory, preserve_nitf_information=args.preserve,
            dem_filename_pattern=args.dem_filename_pattern, dem_type=args.dem_type, geoid_file=args.geoid_file)
