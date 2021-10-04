"""
Create kmz products based on SICD type reader.

For a basic help on the command-line, check

>>> python -m sarpy.utils.create_kmz --help

"""

import argparse
import logging
import os

from sarpy.io.complex.converter import open_complex
from sarpy.io.product.kmz_product_creation import create_kmz_view

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create derived product is SIDD format from a SICD type file.",
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
        help='Path to the output directory where the product file(s) will be created.\n'
             'This directory MUST exist.\n'
             '* Depending on the input file, multiple product files may be produced.\n'
             '* The name for the ouput file(s) will be chosen based on CoreName and\n '
             '  transmit/collect polarization.\n')
    parser.add_argument(
        '-s', '--size', default=3072, type=int, help='Maximum size for the interpolated image, put -1 for full size')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose (level="INFO") logging?')

    args = parser.parse_args()

    if args.verbose:
        logger = logging.getLogger('sarpy')
        logger.setLevel('INFO')

    reader = open_complex(args.input_file)
    file_stem = os.path.splitext(os.path.split(args.input_file)[1])[0]
    pixel_limit = None if args.size == -1 else args.size
    create_kmz_view(reader, args.output_directory, pixel_limit=pixel_limit, file_stem='View-{}'.format(file_stem))
