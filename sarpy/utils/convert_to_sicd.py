"""
Script for converting to SICD format
"""

import argparse
from sarpy.io.complex.converter import conversion_utility

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def convert(input_file, output_dir):
    conversion_utility(input_file, output_dir)


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

    args = parser.parse_args()
    convert(args.input_file, args.output_directory)
