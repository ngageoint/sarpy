"""
Create products based on SICD type reader.

For a basic help on the command-line, check

>>> python -m sarpy.utils.sicd_to_sidd --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

import argparse
import logging
import pathlib
import tempfile

from sarpy.utils import sicd_sidelobe_control, create_product
from sarpy.processing.sicd.spectral_taper import Taper
import sarpy.visualization.remap as remap


def main(args=None):
    remap_names = [r[0] for r in remap.get_remap_list()]

    parser = argparse.ArgumentParser(
        description="Create derived product in SIDD format from a SICD type file.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'input_file', metavar='input_file', help='Input SICD file')
    parser.add_argument(
        'output_directory', metavar='output_directory', help='SIDD Output directory')
    parser.add_argument(
        '-t', '--type', default='detected', choices=['detected', 'csi', 'dynamic'],
        help="The type of derived product. (default: %(default)s)")
    parser.add_argument(
        '-m', '--method', default='nearest', choices=['nearest', ]+['spline_{}'.format(i) for i in range(1, 6)],
        help="The projection interpolation method. (default: %(default)s)")
    sicd_sidelobe_control.window_args_parser(parser)
    parser.add_argument(
        '-r', '--remap', default=remap_names[0], choices=remap_names,
        help="The pixel value remap function. (default: %(default)s)")
    parser.add_argument(
        '--version', default=3, type=int, choices=[1, 2, 3],
        help="The version of the SIDD standard used. (default: %(default)d)")
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose (level="INFO") logging?')
    parser.add_argument(
        '-s', '--sicd', action='store_true', help='Include the SICD structure in the SIDD?')

    args = parser.parse_args(args)

    level = 'INFO' if args.verbose else 'WARNING'
    logging.basicConfig(level=level)
    logger = logging.getLogger('sarpy')
    logger.setLevel(level)

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = pathlib.Path(tempdir)

        input_sicd = args.input_file
        output_sidd_dir = args.output_directory

        # Apply a spectral taper window to the SICD, if necessary
        if args.window is not None:
            windowed_sicd = str(tempdir_path / 'windowed_sicd.nitf')
            sicd_window_args = [input_sicd, windowed_sicd, '--window', args.window]
            if args.pars:
                sicd_window_args += ['--pars'] + args.pars
            sicd_sidelobe_control.main(sicd_window_args)
            input_sicd = windowed_sicd

        # Convert the SICD to the specified SIDD product
        sidd_product_args = [input_sicd, output_sidd_dir, '--type', args.type,
                             '--remap', args.remap, '--method', args.method, '--version', str(args.version)]
        sidd_product_args += ['--verbose'] if args.verbose else []
        sidd_product_args += ['--sicd'] if args.sicd else []

        create_product.main(sidd_product_args)


if __name__ == '__main__':
    main()    # pragma: no cover
