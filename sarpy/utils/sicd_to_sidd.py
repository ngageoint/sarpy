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

import numpy as np

from sarpy.io.complex.sicd import SICDReader
from sarpy.utils import sicd_sidelobe_control, create_product
from sarpy.processing.sicd.spectral_taper import Taper
import sarpy.visualization.remap as remap


def main(args=None):
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
        '-r', '--remap', default='gdm', choices=['gdm', ] + remap.get_remap_names(),
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

        # Register new remap functions that use the global statistics of the SICD.
        with SICDReader(input_sicd) as reader:
            amp = np.abs(reader[:, :])

            if args.remap == 'density':
                remap.register_remap(remap.Density(override_name='density', bit_depth=8, max_output_value=255,
                                                   data_mean=float(np.mean(amp))), overwrite=True)
            elif args.remap == 'high_contrast':
                remap.register_remap(remap.High_Contrast(override_name='high_contrast', bit_depth=8, max_output_value=255,
                                                         data_mean=float(np.mean(amp))), overwrite=True)
            elif args.remap == 'brighter':
                remap.register_remap(remap.Brighter(override_name='brighter', bit_depth=8, max_output_value=255,
                                                    data_mean=float(np.mean(amp))), overwrite=True)
            elif args.remap == 'darker':
                remap.register_remap(remap.Darker(override_name='darker', bit_depth=8, max_output_value=255,
                                                  data_mean=float(np.mean(amp))), overwrite=True)
            elif args.remap == 'pedf':
                remap.register_remap(remap.PEDF(override_name='darker', bit_depth=8, max_output_value=255,
                                                data_mean=float(np.mean(amp))), overwrite=True)
            elif args.remap == 'gdm':
                graze_deg = reader.sicd_meta.SCPCOA.GrazeAng
                slope_deg = reader.sicd_meta.SCPCOA.SlopeAng

                # If there is any non-uniform spectral weighting then GDM will choose parameters as if
                # Taylor weighting was applied, instead of choosing the parameters for Uniform weighting.
                if args.window is None:
                    row_wgt = [1] if reader.sicd_meta.Grid.Row.WgtFunct is None else reader.sicd_meta.Grid.Row.WgtFunct
                    col_wgt = [1] if reader.sicd_meta.Grid.Col.WgtFunct is None else reader.sicd_meta.Grid.Col.WgtFunct
                    is_uniform = np.allclose(row_wgt, row_wgt[0]) and np.allclose(col_wgt, col_wgt[0])
                    weighting = 'UNIFORM' if is_uniform else 'TAYLOR'
                else:
                    weighting = args.window

                remap.register_remap(remap.GDM(override_name='gdm', bit_depth=8, max_output_value=255,
                                               weighting=weighting, graze_deg=graze_deg, slope_deg=slope_deg,
                                               data_mean=float(np.mean(amp)), data_median=float(np.median(amp))),
                                     overwrite=True)
            elif args.remap == 'linear':
                remap.register_remap(remap.Linear(override_name='linear', bit_depth=8, max_output_value=255,
                                                  min_value=float(np.min(amp)),  max_value=float(np.max(amp))),
                                     overwrite=True)
            elif args.remap == 'log':
                remap.register_remap(remap.Logarithmic(override_name='log', bit_depth=8, max_output_value=255,
                                                       min_value=float(np.min(amp)),  max_value=float(np.max(amp))),
                                     overwrite=True)
            else:
                logger.warning(f'Remap function "{args.remap}" might not be using global statistics.')

            del amp

        # Convert the SICD to the specified SIDD product
        sidd_product_args = [input_sicd, output_sidd_dir, '--type', args.type,
                             '--remap', args.remap, '--method', args.method, '--version', str(args.version)]
        sidd_product_args += ['--verbose'] if args.verbose else []
        sidd_product_args += ['--sicd'] if args.sicd else []

        create_product.main(sidd_product_args)


if __name__ == '__main__':
    main()    # pragma: no cover
