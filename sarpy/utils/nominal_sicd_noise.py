"""
Add a nominal noise polynomial to a sicd.

For a basic help on the command-line, check

>>> python -m sarpy.utils.nominal_sicd_noise --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
import argparse

import numpy

import sarpy
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.complex.converter import conversion_utility
from sarpy.io.complex.sicd_elements.Radiometric import NoiseLevelType_

logger = logging.getLogger(__name__)


def nominal_sicd_noise(
        input_reader, out_directory, output_file=None, noise_db_value=-16.0, override=False,
        check_existence=True, check_older_version=False, preserve_nitf_information=False):
    """
    Create a sicd with the nominal noise value.

    Parameters
    ----------
    input_reader : str|SICDReader
    out_directory : str
        The directory of the output.
    output_file : None|str
        If `None`, then the name will mirror the original with noise details appended.
    noise_db_value : int|float
        The estimate for nesz in decibels.
    override : bool
        If a NoisePoly is already populated, should we override the value? If `False`
        and a NoisePoly is already populated, then an exception will be raised.
    check_existence : bool
        Check for the existence of the file before overwriting?
    check_older_version : bool
        Try to use a less recent version of SICD (1.1), for possible application
        compliance issues?
    preserve_nitf_information : bool
        Try to preserve some of the original NITF information?
    """

    if isinstance(input_reader, str):
        input_reader = SICDReader(input_reader)
    if not isinstance(input_reader, SICDReader):
        raise TypeError('We require that the input is a SICD reader or path to a sicd file.')

    noise_db_value = float(noise_db_value)
    if noise_db_value > -4:
        logger.warning(
            'The noise estimate should be provided in dB,\n\t'
            'and the provided value is `{}`.\n\t'
            'Maybe this is an error?'.format(noise_db_value))

    if output_file is None:
        fname = os.path.split(input_reader.file_name)[1]
        fstem, fext = os.path.splitext(fname)
        fstem += '_{}dB_noise'.format(int(noise_db_value))
        fname = fstem + fext
    else:
        fname = os.path.split(output_file)[1]

    sicd = input_reader.sicd_meta
    if sicd.Radiometric is None:
        raise ValueError(
            'The provided sicd does not contain any radiometric information,\n\t'
            'and SigmaZeroSFPoly is required to proceed')

    new_noise = NoiseLevelType_(
        NoisePoly=[[noise_db_value - 10 * numpy.log10(sicd.Radiometric.SigmaZeroSFPoly[0, 0]), ]],
        NoiseLevelType='ABSOLUTE')

    if sicd.Radiometric.NoiseLevel is None:
        original_noise = None
    else:
        if not override:
            raise ValueError(
                'The provided sicd already contains radiometric noise information,\n\t'
                'set override=True to replace the value')
        original_noise = sicd.Radiometric.NoiseLevel.copy()
    sicd.Radiometric.NoiseLevel = new_noise

    conversion_utility(
        input_reader, out_directory, output_files=fname,
        check_existence=check_existence,
        check_older_version=check_older_version,
        preserve_nitf_information=preserve_nitf_information)

    # return to the original noise information, so we haven't modified any in memory reader information
    sicd.Radiometric.NoiseLevel = original_noise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subset SICD file.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'input_file', metavar='input_file',
        help='Path input sicd data file.')
    parser.add_argument(
        'output_directory', metavar='output_directory',
        help='Path to the output directory. This directory MUST exist.')
    parser.add_argument(
        '--override', action='store_true',
        help='Override any present Noise polynomial?')
    parser.add_argument(
        '-o', '--output_file', default=None, type=str)
    parser.add_argument(
        '-n', '--noise', default=-16.0, type=float,
        help='Nominal nesz noise value in decibels')
    parser.add_argument(
        '-p', '--preserve', action='store_true',
        help='Try to preserve any NITF information?\n'
             'This only applies in the event that the file being read is a NITF')
    parser.add_argument(
        '-w', '--overwrite', action='store_true',
        help='Overwrite output file, if it already exists?')
    parser.add_argument(
        '--older', action='store_true',
        help='Try to use a less recent version of SICD (1.1),\n'
             'for possible application compliance issues?')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose (level="INFO") logging?')

    args = parser.parse_args()

    level = 'INFO' if args.verbose else 'WARNING'
    logging.basicConfig(level=level)
    logger = logging.getLogger('sarpy')
    logger.setLevel(level)

    nominal_sicd_noise(
        args.input_file, args.output_directory, output_file=args.output_file,
        noise_db_value=args.noise,
        override=args.override,
        check_existence=not args.overwrite,
        check_older_version=args.older,
        preserve_nitf_information=args.preserve)
