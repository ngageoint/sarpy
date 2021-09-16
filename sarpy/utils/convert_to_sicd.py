"""
Convert from complex SAR image format to SICD format.

For a basic help on the command-line, check

>>> python -m sarpy.utils.convert_to_sicd --help

"""

import argparse
from sarpy.io.complex.converter import conversion_utility

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def convert(input_file, output_dir, preserve_nitf_information=False):
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
    """

    conversion_utility(input_file, output_dir, preserve_nitf_information=preserve_nitf_information)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert to SICD format.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'input_file', metavar='input_file',
        help='Path input data file, or directory for radarsat, RCM, or sentinel.\n'
             '* For radarsat or RCM, this can be the product.xml file, or parent directory\n'
             '  of product.xml or metadata/product.xml.\n'
             '* For sentinel, this can be the manifest.safe file, or parent directory of\n'
             '  manifest.safe.\n')
    parser.add_argument(
        'output_directory', metavar='output_directory',
        help='Path to the output directory. This directory MUST exist.\n'
             '* Depending on the input details, multiple SICD files may be produced.\n'
             '* The name for the ouput file(s) will be chosen based on CoreName and\n '
             '  transmit/collect polarization.\n')
    parser.add_argument(
        '-p', '--preserve', action='store_true',
        help='Try to preserve any NITF information?\n'
             'This only applies in the event that the file being read is a NITF')

    args = parser.parse_args()
    convert(args.input_file, args.output_directory, preserve_nitf_information=args.preserve)
