# -*- coding: utf-8 -*-
"""
This package contains the elements for interpreting complex radar data in a variety of formats.
For non-SICD files, the radar metadata will be converted to something compatible with the SICD
standard, to the extent feasible.

It also permits converting complex data from any form which can be read to a file or files in
SICD or SIO format.
"""

import os
import sys
import pkgutil
import numpy
import logging
from typing import List, Tuple

from .base import BaseReader
from .sicd import SICDWriter
from .sio import SIOWriter
from .sicd_elements.SICD import SICDType


integer_types = (int, )
int_func = int
if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    int_func = long  # to accommodate for 32-bit python 2
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)


_writer_types = {'SICD': SICDWriter, 'SIO': SIOWriter}

__classification__ = "UNCLASSIFIED"
__author__ = ("Wade Schwartzkopf", "Thomas McCullough")


def open(file_name):
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    BaseReader

    Raises
    ------
    IOError
    """

    # TODO: can we refactor this name?

    if not os.path.exists(file_name) or not os.path.isfile(file_name):
        raise IOError('File {} does not exist.'.format(file_name))

    # Get a list of all the modules that describe SAR complex image file
    # formats.  For this to work, all of the file format handling modules
    # must be in the top level of this package.
    mod_iter = pkgutil.iter_modules(__path__, __name__ + '.')
    module_names = [name for loader, name, ispkg in mod_iter if not ispkg]
    modules = [sys.modules[names] for names in module_names if __import__(names)]
    # Determine file format and return the proper file reader object
    for current_mod in modules:
        if hasattr(current_mod, 'is_a'):  # Make sure its a file format handling module
            reader = current_mod.is_a(file_name)
            if reader is not None:
                return reader
    # If for loop completes, no matching file format was found.
    raise IOError('Unable to determine complex image format.')


def convert(input_file, output_file, frames=None, output_format='SICD',
            row_limits=None, column_limits=None, max_block_size=2**26):
    """
    Copy SAR complex data to a file of the specified format.

    Parameters
    ----------
    input_file : str|BaseReader
       Reader instance, or the name of file to convert.
    output_file : str|List[str]
       The name of the output file, or list of output files matching `frames`.
    frames : None|int|list
       Set of frames to convert. Default is all.
    output_format : str
       The output file format to write, from {'SICD', 'SIO'}, optional.  Default is SICD.
    row_limits : None|Tuple[int, int]|List[Tuple[int, int]]
       Rows start/stop. Default is all.
    column_limits : None|Tuple[int, int]|List[Tuple[int, int]]
       Columns start/stop. Default is all.
    max_block_size : int
        (nominal) maximum block size in bytes. Minimum value is 2**20 (1 MB).

    Returns
    -------
    None
    """

    def validate_lims(lims, typ):
        def validate_entry(st, ed, shap, i_fr):
            if not ((0 <= st < shap[ind]) and (st < ed <= shap[ind])):
                raise ValueError('{}_limits is {}, and frame {} has shape {}'.format(typ, lims, i_fr, shap))

        ind = 0 if typ == 'row' else 1

        if lims is None:
            return tuple((0, shp[ind]) for shp in sizes)
        else:
            o_lims = numpy.array(lims, dtype=numpy.int64)
            t_lims = []
            if len(o_lims.shape) == 1:
                if o_lims.shape[0] != 2:
                    raise ValueError(
                        'row{}_limits must be of the form (<start>, <end>), got {}'.format(typ, lims))
                t_start = int_func(o_lims[0])
                t_end = int_func(o_lims[1])
                for i_frame, shp in zip(frames, sizes):
                    validate_entry(t_start, t_end, shp, i_frame)
                    t_lims.append((t_start, t_end))
            else:
                if o_lims.shape[0] != len(frames):
                    raise ValueError(
                        '{0:s}_limits must either be of the form (<start>, <end>) applied to all frames, '
                        'or a collection of such of the same length as frames. '
                        'Got len({0:s}_limits) = {1:d} and len(frames) = {2:d}'.format(typ, o_lims.shape[0], len(frames)))
                for entry, i_frame, shp in zip(o_lims, frames, sizes):
                    t_start = int_func(entry[0])
                    t_end = int_func(entry[1])
                    validate_entry(t_start, t_end, shp, i_frame)
                    t_lims.append((t_start, t_end))
            return tuple(t_lims)

    def update_sicd(t_sicd, rows, cols, t_size):
        # type: (SICDType, Tuple[int, int], Tuple[int, int], Tuple[int, int]) -> SICDType
        o_sicd = t_sicd.copy()
        if rows != (0, t_size[0]) or cols != (0, t_size[1]):
            o_sicd.ImageData.NumRows = rows[1] - rows[0]
            o_sicd.ImageData.NumCols = cols[1] - cols[0]
            o_sicd.ImageData.FirstRow = t_sicd.ImageData.FirstRow + rows[0]
            o_sicd.ImageData.FirstCol = t_sicd.ImageData.FirstCol + cols[0]
            o_sicd.define_geo_image_corners(override=True)
        return o_sicd

    def bytes_per_row(t_sicd):
        # type: (SICDType) -> int
        pixel_type = t_sicd.ImageData.PixelType
        cols = t_sicd.ImageData.NumCols
        if pixel_type == 'RE32F_IM32F':
            return 8*cols
        elif pixel_type == 'RE16I_IM16I':
            return 4*cols
        elif pixel_type == 'AMP8I_PHS8I':
            return 2*cols
        else:
            return 8*cols

    def rows_per_block(t_sicd):
        # type: (SICDType) -> int
        bpr = bytes_per_row(t_sicd)
        return max(1, int_func(max_block_size/bpr))

    if output_format is None:
        output_format = 'SICD'
    output_format = output_format.upper()
    if output_format not in ['SICD', 'SIO']:
        raise ValueError('Got unexpected output_format {}'.format(output_format))
    writer_type = _writer_types[output_format]

    if isinstance(input_file, str):
        reader = open(input_file)
    elif isinstance(input_file, BaseReader):
        reader = input_file
    else:
        raise ValueError(
            'input_file is expected to be a file name or Reader instance. Got {}'.format(type(input_file)))
    sicds = reader.sicd_meta
    sizes = reader.data_size

    if isinstance(sicds, SICDType):
        sicds = (sicds, )
        sizes = (sizes, )
        frames = (0, )

    # check that frames is valid
    if frames is None:
        frames = tuple(range(len(sicds)))
    if isinstance(frames, int):
        frames = (frames, )
    if not isinstance(frames, tuple):
        frames = tuple(int_func(entry) for entry in frames)
    if len(frames) == 0:
        raise ValueError('The list of frames is empty.')
    o_frames = []
    for frame in frames:
        index = int_func(frame)
        if not (0 <= index < len(sicds)):
            raise ValueError(
                'Got a frames entry {}, but it must be between 0 and {}'.format(index, len(sicds)))
        o_frames.append(index)
    frames = tuple(o_frames)

    # construct output_files list
    if isinstance(output_file, str):
        if len(sicds) == 1:
            output_file = [output_file, ]
        else:
            digits = int_func(numpy.ceil(numpy.log10(len(sicds))))
            frm_str = '{0:s}-1{:0' + str(digits) + 'd}{2:s}'
            fstem, fext = os.path.splitext(output_file)
            o_files = []
            for index in frames:
                o_files.append(frm_str.format(fstem, index, fext))
            output_file = tuple(o_files)

    # construct validated row/column_limits
    row_limits = validate_lims(row_limits, 'row')
    column_limits = validate_lims(column_limits, 'column')
    # validate max_block_size
    max_block_size = int_func(max_block_size)
    if max_block_size < 2**20:
        max_block_size = 2**20

    for o_file, frame, row_lims, col_lims in zip(output_file, frames, row_limits, column_limits):
        logging.info('Converting frame {} from file {} to file {}'.format(frame, input_file, o_file))
        this_size = sizes[frame]
        this_sicd = update_sicd(sicds[frame], row_lims, col_lims, this_size)
        rpb = rows_per_block(this_sicd)
        block_start = row_lims[0]
        with writer_type(o_file, this_sicd) as writer:
            while block_start < row_lims[1]:
                block_end = min(block_start + rpb, row_lims[1])
                data = reader.read_chip((block_start, block_end, 1), None, index=frame)
                writer.write_chip(data, start_indices=(block_start-row_lims[0], 0))
                logging.info('Done writing block {}-{} to file {}'.format(block_start, block_end, o_file))
                block_start = block_end
