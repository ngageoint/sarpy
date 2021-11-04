"""
Create products based on SICD type reader.

For a basic help on the command-line, check

>>> python -m sarpy.utils.create_product --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import argparse
import logging

import sarpy
from sarpy.io.complex.converter import open_complex
from sarpy.processing.ortho_rectify import BivariateSplineMethod, NearestNeighborMethod
from sarpy.io.product.sidd_product_creation import create_detected_image_sidd, \
    create_csi_sidd, create_dynamic_image_sidd


def _parse_method(method):
    if method.startswith('spline_'):
        return int(method[-1])
    else:
        return 'nearest'


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
        '-t', '--type', default='detected', choices=['detected', 'csi', 'dynamic'],
        help="The type of derived product.")
    parser.add_argument(
        '-m', '--method', default='nearest', choices=['nearest', ]+['spline_{}'.format(i) for i in range(1, 6)],
        help="The interpolation method.")
    parser.add_argument(
        '--version', default=2, type=int, choices=[1, 2],
        help="The version of the SIDD standard used.")
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose (level="INFO") logging?')
    parser.add_argument(
        '-s', '--sicd', action='store_true', help='Include the SICD structure in the SIDD?')

    args = parser.parse_args()

    level = 'INFO' if args.verbose else 'WARNING'
    logging.basicConfig(level=level)
    logger = logging.getLogger('sarpy')
    logger.setLevel(level)

    reader = open_complex(args.input_file)
    degree = _parse_method(args.method)
    for i, sicd in enumerate(reader.get_sicds_as_tuple()):
        if isinstance(degree, int):
            ortho_helper = BivariateSplineMethod(reader, index=i, row_order=degree, col_order=degree)
        else:
            ortho_helper = NearestNeighborMethod(reader, index=i)
        if args.type == 'detected':
            create_detected_image_sidd(ortho_helper, args.output_directory, version=args.version, include_sicd=args.sicd)
        elif args.type == 'csi':
            create_csi_sidd(ortho_helper, args.output_directory, version=args.version, include_sicd=args.sicd)
        elif args.type == 'dynamic':
            create_dynamic_image_sidd(ortho_helper, args.output_directory, version=args.version, include_sicd=args.sicd)
        else:
            raise ValueError('Got unhandled type {}'.format(args.type))
