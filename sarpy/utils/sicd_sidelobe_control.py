"""
Apply or remove an impulse response side lobe control window function.

For a basic help on the command-line, check

>>> python -m sarpy.utils.sicd_sidelobe_control --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "Valkyrie Systems Corporation"

import argparse
import logging

import numpy as np

from sarpy.io.complex.sicd import SICDReader, SICDWriter, validate_sicd_for_writing
from sarpy.processing.sicd.spectral_taper import Taper, apply_spectral_taper


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Remove and/or apply a sidelobe control spectral taper window to a SICD file.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'input_file', metavar='input_file', help='Path to the input data SICD file.')
    parser.add_argument(
        'output_file', metavar='output_file', help='Path to the output SICD file.\n')
    window_args_parser(parser)
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose (level="INFO") logging?')

    args = parser.parse_args(args)

    window = 'UNIFORM' if args.window is None else args.window.upper()
    default_pars = Taper().default_pars

    # Convert the list of par values into a dict of pars {name: value}.
    pars = {p: args.pars[n] for n, p in enumerate(default_pars.get(window, {}))}

    level = 'INFO' if args.verbose else 'WARNING'
    logging.basicConfig(level=level)
    logger = logging.getLogger('sarpy')
    logger.setLevel(level)

    reader = SICDReader(args.input_file)

    taper = Taper(window, pars)

    cdata, mdata = apply_spectral_taper(reader, taper)

    # Write the results to a new SICD file.
    writer = SICDWriter(
        file_object=args.output_file,
        sicd_meta=validate_sicd_for_writing(mdata),
        check_existence=False
    )
    writer.write(cdata.astype(np.complex64))
    writer.close()


def window_args_parser(parser):
    def pars_text(pars):
        prefix = 'takes 1 parameter: ' if len(pars) == 1 else f'takes {len(pars)} parameters: '
        return prefix + ', '.join([f'"{p.lower()}"' for p in pars.keys()])

    window_name_choices = [w.lower() for w in Taper().default_pars.keys()]
    window_pars_choices = [f'  "{w.lower()}": {pars_text(p)}' for w, p in Taper().default_pars.items() if p]
    parser.add_argument(
        '-w', '--window', type=str.lower, choices=window_name_choices,
        help='The name of the window function.  Acceptable (case insensitive) names are:\n'
             '  "uniform": This is a flat spectral taper (i.e., no spectral taper).\n'
             '  "hamming": Hamming window taper (0.54 + 0.46 * cos(2*pi*n/M))\n'
             '  "hanning" | "hann":  Hann window taper (0.5 + 0.5 * cos(2*pi*n/M))\n'
             '  "general_hamming": Raised cosine window (alpha + (1 - alpha) * cos(2*pi*n/M)), default (alpha = 0.5)\n'
             '  "kaiser": Kaiser-Bessel window (I0[beta * sqrt(1 - (2*n/M - 1)**2)] / I0[beta]) default (beta = 14)\n'
             '  "taylor": Taylor window: Default (nbar = 4, sll = -30)\n'
    )
    parser.add_argument(
        '-p', '--pars', nargs='+',
        help=("One or more parameter values to modify the window characteristics.\n" +
              '\n'.join(window_pars_choices +
                        ["For example:",
                         "  --window general_hamming --pars 0.6",
                         "  --window taylor          --pars 5 -35.0",
                         "  --window kaiser          --pars 15"]
                        )
              )
    )


if __name__ == '__main__':
    main()    # pragma: no cover
