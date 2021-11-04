"""
Create a chip (subimage) of a SICD image.

For a basic help on the command-line, check

>>> python -m sarpy.utils.chip_sicd --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "John Gorman"

import os
import argparse
from typing import Tuple
import logging

import sarpy
from sarpy.io.complex.converter import conversion_utility
from sarpy.io.complex.sicd import SICDReader


def _verify_limits(limits):
    """
    Helper function to verify that the row/column limits are sensible.

    Parameters
    ----------
    limits : None|tuple|list

    Returns
    -------
    None|tuple
    """

    if limits is None:
        return limits

    temp_limits = [int(entry) for entry in limits]
    if len(temp_limits) != 2:
        raise ValueError('Got unexpected limits `{}`'.format(limits))
    if not (0 <= temp_limits[0] < temp_limits[1]):
        raise ValueError('Got unexpected limits `{}`'.format(limits))
    return temp_limits[0], temp_limits[1]


def create_chip(input_reader, out_directory, output_file=None, row_limits=None, col_limits=None,
                check_existence=True, check_older_version=False, preserve_nitf_information=False):
    """
    Create a chip of the given sicd file. At least one of `row_limits` and
    `col_limits` must be provided.

    Parameters
    ----------
    input_reader : str|SICDReader
    out_directory : str
        The directory of the output.
    output_file : None|str
        If `None`, then the name will mirror the original with row/col details appended.
    row_limits : None|Tuple[int, int]
        The limits for the rows, relative to this actual image, to be included.
    col_limits : None|Tuple[int, int]
        The limits for the columns, relative to this actual image, to be included.
    check_existence : bool
        Check for the existence of the file before overwriting?
    check_older_version : bool
        Try to use a less recent version of SICD (1.1), for possible application compliance issues?
    preserve_nitf_information : bool
        Try to preserve some of the original NITF information?
    """

    def get_suffix(limits, shift):
        if limits is None:
            return 'all'
        else:
            return '{0:d}-{1:d}'.format(shift+limits[0], shift+limits[1])

    if isinstance(input_reader, str):
        input_reader = SICDReader(input_reader)
    if not isinstance(input_reader, SICDReader):
        raise TypeError('We require that the input is a SICD reader or path to a sicd file.')

    row_limits = _verify_limits(row_limits)
    col_limits = _verify_limits(col_limits)

    if row_limits is None and col_limits is None:
        raise ValueError('At least one of row_limits and col_limits must be provided.')

    if output_file is None:
        fname = os.path.split(input_reader.file_name)[1]
        fstem, fext = os.path.splitext(fname)
        fstem += '_{}_{}'.format(
            get_suffix(row_limits, input_reader.sicd_meta.ImageData.FirstRow),
            get_suffix(col_limits, input_reader.sicd_meta.ImageData.FirstCol))
        fname = fstem + fext
    else:
        fname = os.path.split(output_file)[1]

    conversion_utility(
        input_reader, out_directory, output_files=fname,
        row_limits=row_limits, column_limits=col_limits,
        check_existence=check_existence,
        check_older_version=check_older_version,
        preserve_nitf_information=preserve_nitf_information)


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
        '-o', '--output_file', default=None, type=str)
    parser.add_argument(
        '-r', '--row_lims', default=None, nargs=2, type=int,
        help='Row limits for chip, integers: row_start row_stop')
    parser.add_argument(
        '-c', '--col_lims', default=None, nargs=2, type=int,
        help='Column limits for chip, integers: col_start col_stop')
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

    create_chip(
        args.input_file, args.output_directory, output_file=args.output_file,
        row_limits=args.row_lims, col_limits=args.col_lims,
        check_existence=not args.overwrite,
        check_older_version=args.older,
        preserve_nitf_information=args.preserve)
