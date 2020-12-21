# -*- coding: utf-8 -*-
"""
This module provide utilities for converting from any complex format that we can
read to SICD or SIO format. The same conversion utility can be used to subset data.
"""

import os
import sys
import pkgutil
from importlib import import_module
import numpy
import logging
from typing import Union, List, Tuple

from sarpy.compliance import int_func
from sarpy.io.general.base import BaseReader
from sarpy.io.complex.sicd import SICDWriter
from sarpy.io.complex.sio import SIOWriter
from sarpy.io.complex.sicd_elements.SICD import SICDType


__classification__ = "UNCLASSIFIED"
__author__ = ("Wade Schwartzkopf", "Thomas McCullough")


###########
# Module variables
_writer_types = {'SICD': SICDWriter, 'SIO': SIOWriter}
_openers = []
_parsed_openers = False


def register_opener(open_func):
    """
    Provide a new opener.

    Parameters
    ----------
    open_func : callable
        This is required to be a function which takes a single argument (file name).
        This function should return a sarpy.io.complex.base.BaseReader instance
        if the referenced file is viable for the underlying type, and None otherwise.

    Returns
    -------
    None
    """

    if not callable(open_func):
        raise TypeError('open_func must be a callable')
    if open_func not in _openers:
        _openers.append(open_func)


def parse_openers():
    """
    Automatically find the viable openers (i.e. :func:`is_a`) in the various modules.

    Returns
    -------

    """

    global _parsed_openers
    if _parsed_openers:
        return
    _parsed_openers = True

    def check_module(mod_name):
        # import the module
        import_module(mod_name)
        # fetch the module from the modules dict
        module = sys.modules[mod_name]
        # see if it has an is_a function, if so, register it
        if hasattr(module, 'is_a'):
            register_opener(module.is_a)

        # walk down any subpackages
        path, fil = os.path.split(module.__file__)
        if not fil.startswith('__init__.py'):
            # there are no subpackages
            return
        for sub_module in pkgutil.walk_packages([path, ]):
            _, sub_module_name, _ = sub_module
            sub_name = "{}.{}".format(mod_name, sub_module_name)
            check_module(sub_name)

    check_module('sarpy.io.complex')


def _define_final_attempt_openers():
    """
    Gets the prioritized list of openers to attempt after regular openers.

    Returns
    -------
    list
    """

    from sarpy.io.complex.other_nitf import final_attempt
    return [final_attempt, ]


def open_complex(file_name):
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

    if not os.path.exists(file_name):
        raise IOError('File {} does not exist.'.format(file_name))
    # parse openers, if not already done
    parse_openers()
    # see if we can find a reader though trial and error
    for opener in _openers:
        reader = opener(file_name)
        if reader is not None:
            return reader
    # check the final attempt openers
    for opener in _define_final_attempt_openers():
        reader = opener(file_name)
        if reader is not None:
            return reader

    # If for loop completes, no matching file format was found.
    raise IOError('Unable to determine complex image format.')


class Converter(object):
    """
    This is a class for conversion (of a single frame) of one complex format to
    SICD or SIO format. Another use case is to create a (contiguous) subset of a
    given complex dataset. **This class is intended to be used as a context manager.**
    """

    __slots__ = ('_reader', '_file_name', '_writer', '_frame', '_row_limits', '_col_limits')

    def __init__(self, reader, output_directory, output_file=None, frame=None, row_limits=None, col_limits=None,
                 output_format='SICD', check_older_version=False):
        """

        Parameters
        ----------
        reader : BaseReader
            The base reader instance.
        output_directory : str
            The output directory. **This must exist.**
        output_file : None|str
            The output file name. If not provided, then `sicd.get_suggested_name(frame)`
            will be used.
        frame : None|int
            The frame (i.e. index into the reader's sicd collection) to convert.
            The default is 0.
        row_limits : None|Tuple[int, int]
           Row start/stop. Default is all.
        col_limits : None|Tuple[int, int]
           Column start/stop. Default is all.
        output_format : str
           The output file format to write, from {'SICD', 'SIO'}.  Default is SICD.
        check_older_version : bool
            Try to use a less recent version of SICD (1.1), for possible application compliance issues?
        """

        if not (os.path.exists(output_directory) and os.path.isdir(output_directory)):
            raise IOError('output directory {} must exist.'.format(output_directory))
        if output_file is None:
            output_file = reader.get_sicds_as_tuple()[frame].get_suggested_name(frame+1)+'_SICD'
        output_path = os.path.join(output_directory, output_file)
        if os.path.exists(output_path):
            raise IOError('The file {} already exists.'.format(output_path))

        # validate the output format and fetch the writer type
        if output_format is None:
            output_format = 'SICD'
        output_format = output_format.upper()
        if output_format not in ['SICD', 'SIO']:
            raise ValueError('Got unexpected output_format {}'.format(output_format))
        writer_type = _writer_types[output_format]

        if isinstance(reader, BaseReader):
            self._reader = reader  # type: BaseReader
        else:
            raise ValueError(
                'reader is expected to be a Reader instance. Got {}'.format(type(reader)))

        # fetch the appropriate sicd instance
        sicds = reader.get_sicds_as_tuple()
        shapes = reader.get_data_size_as_tuple()
        if frame is None:
            self._frame = 0
        else:
            self._frame = int_func(frame)
        if not (0 <= self._frame < len(sicds)):
            raise ValueError(
                'Got a frame {}, but it must be between 0 and {}'.format(frame, len(sicds)))
        this_sicd = sicds[self._frame]
        this_shape = shapes[self._frame]

        # validate row_limits and col_limits
        if row_limits is None:
            row_limits = (0, this_shape[0])
        else:
            row_limits = (int_func(row_limits[0]), int_func(row_limits[1]))
            if not ((0 <= row_limits[0] < this_shape[0]) and (row_limits[0] < row_limits[1] <= this_shape[0])):
                raise ValueError(
                    'Entries of row_limits must be monotonically increasing '
                    'and in the range [0, {}]'.format(this_shape[0]))
        if col_limits is None:
            col_limits = (0, this_shape[1])
        else:
            col_limits = (int_func(col_limits[0]), int_func(col_limits[1]))
            if not ((0 <= col_limits[0] < this_shape[1]) and (col_limits[0] < col_limits[1] <= this_shape[1])):
                raise ValueError(
                    'Entries of col_limits must be monotonically increasing '
                    'and in the range [0, {}]'.format(this_shape[1]))
        self._row_limits = row_limits  # type: Tuple[int, int]
        self._col_limits = col_limits  # type: Tuple[int, int]
        # redefine our sicd, as necessary
        this_sicd = self._update_sicd(this_sicd, this_shape)
        # set up our writer
        self._file_name = output_path
        self._writer = writer_type(output_path, this_sicd, check_older_version=check_older_version)

    def _update_sicd(self, sicd, t_size):
        # type: (SICDType, Tuple[int, int]) -> SICDType
        o_sicd = sicd.copy()
        if self._row_limits != (0, t_size[0]) or self._col_limits != (0, t_size[1]):
            o_sicd.ImageData.NumRows = self._row_limits[1] - self._row_limits[0]
            o_sicd.ImageData.NumCols = self._col_limits[1] - self._col_limits[0]
            o_sicd.ImageData.FirstRow = sicd.ImageData.FirstRow + self._row_limits[0]
            o_sicd.ImageData.FirstCol = sicd.ImageData.FirstCol + self._col_limits[0]
            o_sicd.define_geo_image_corners(override=True)
        return o_sicd

    def _get_rows_per_block(self, max_block_size):
        pixel_type = self._writer.sicd_meta.ImageData.PixelType
        cols = int_func(self._writer.sicd_meta.ImageData.NumCols)
        bytes_per_row = 8*cols
        if pixel_type == 'RE32F_IM32F':
            bytes_per_row = 8*cols
        elif pixel_type == 'RE16I_IM16I':
            bytes_per_row = 4*cols
        elif pixel_type == 'AMP8I_PHS8I':
            bytes_per_row = 2*cols
        return max(1, int_func(round(max_block_size/bytes_per_row)))

    @property
    def writer(self):  # type: () -> Union[SICDWriter, SIOWriter]
        """SICDWriter|SIOWriter: The writer instance."""
        return self._writer

    def write_data(self, max_block_size=None):
        r"""
        Assuming that the desired changes have been made to the writer instance
        nitf header tags, write the data.

        Parameters
        ----------
        max_block_size : None|int
            (nominal) maximum block size in bytes. Minimum value is :math:`2^{20} = 1~\text{MB}`.
            Default value is :math:`2^{26} = 64~\text{MB}`.

        Returns
        -------
        None
        """

        # validate max_block_size
        if max_block_size is None:
            max_block_size = 2**26
        else:
            max_block_size = int_func(max_block_size)
            if max_block_size < 2**20:
                max_block_size = 2**20

        # now, write the data
        rows_per_block = self._get_rows_per_block(max_block_size)
        block_start = self._row_limits[0]
        while block_start < self._row_limits[1]:
            block_end = min(block_start + rows_per_block, self._row_limits[1])
            data = self._reader[block_start:block_end, self._col_limits[0]:self._col_limits[1], self._frame]
            self._writer.write_chip(data, start_indices=(block_start - self._row_limits[0], 0))
            logging.info('Done writing block {}-{} to file {}'.format(block_start, block_end, self._file_name))
            block_start = block_end

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None:
            self._writer.close()
        else:
            logging.error(
                'The {} file converter generated an exception during processing. The file {} may be '
                'only partially generated and corrupt.'.format(self.__class__.__name__, self._file_name))
            # The exception will be reraised.
            # It's unclear how any exception could be caught.


def conversion_utility(
        input_file, output_directory, output_files=None, frames=None, output_format='SICD',
        row_limits=None, column_limits=None, max_block_size=None, check_older_version=False):
    """
    Copy SAR complex data to a file of the specified format.

    Parameters
    ----------
    input_file : str|BaseReader
       Reader instance, or the name of file to convert.
    output_directory : str
        The output directory. **This must exist.**
    output_files : None|str|List[str]
       The name of the output file(s), or list of output files matching `frames`.
       If not provided, then `sicd.get_suggested_name(frame)` will be used.
    frames : None|int|list
       Set of frames to convert. Default is all.
    output_format : str
       The output file format to write, from {'SICD', 'SIO'}, optional.  Default is SICD.
    row_limits : None|Tuple[int, int]|List[Tuple[int, int]]
       Rows start/stop. Default is all.
    column_limits : None|Tuple[int, int]|List[Tuple[int, int]]
       Columns start/stop. Default is all.
    max_block_size : None|int
        (nominal) maximum block size in bytes. Passed through to the Converter class.
    check_older_version : bool
        Try to use a less recent version of SICD (1.1), for possible application compliance issues?

    Returns
    -------
    None
    """

    def validate_lims(lims, typ):
        # type: (Union[None, tuple, list, numpy.ndarray], str) -> Tuple[Tuple[int, int], ...]
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

    if isinstance(input_file, str):
        reader = open_complex(input_file)
    elif isinstance(input_file, BaseReader):
        reader = input_file
    else:
        raise ValueError(
            'input_file is expected to be a file name or Reader instance. Got {}'.format(type(input_file)))

    if not (os.path.exists(output_directory) and os.path.isdir(output_directory)):
        raise IOError('output directory {} must exist.'.format(output_directory))

    sicds = reader.get_sicds_as_tuple()
    sizes = reader.get_data_size_as_tuple()

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

    # assign SUGGESTED_NAME to each sicd
    for frame in frames:
        sicd = sicds[frame]
        suggested_name = sicd.get_suggested_name(frame+1)+'_SICD'
        if suggested_name is None and sicd.CollectionInfo.CoreName is not None:
            suggested_name = sicd.CollectionInfo.CoreName+'{}_SICD'.format(frame)
        if suggested_name is None:
            suggested_name = 'Unknown{}_SICD'.format(frame)
        if hasattr(sicd, '_NITF') and isinstance(sicd._NITF, dict):
            sicd._NITF['SUGGESTED_NAME'] = suggested_name
        else:
            sicd._NITF = {'SUGGESTED_NAME': suggested_name}
    # construct output_files list
    if output_files is None:
        output_files = [sicds[frame]._NITF['SUGGESTED_NAME']+'.nitf' for frame in frames]
    elif isinstance(output_files, str):
        if len(sicds) == 1:
            output_files = [output_files, ]
        else:
            digits = int_func(numpy.ceil(numpy.log10(len(sicds))))
            frm_str = '{0:s}-{1:0' + str(digits) + 'd}{2:s}'
            fstem, fext = os.path.splitext(output_files)
            o_files = []
            for index in frames:
                o_files.append(frm_str.format(fstem, index, fext))
            output_files = tuple(o_files)

    if len(output_files) != len(frames):
        raise ValueError('The lengths of frames and output_files must match.')
    if len(set(output_files)) != len(output_files):
        raise ValueError(
            'Entries in output_files (possibly constructed) must be unique, got {} for frames {}'.format(output_files, frames))

    # construct validated row/column_limits
    row_limits = validate_lims(row_limits, 'row')
    column_limits = validate_lims(column_limits, 'column')

    for o_file, frame, row_lims, col_lims in zip(output_files, frames, row_limits, column_limits):
        logging.info('Converting frame {} from file {} to file {}'.format(frame, input_file, o_file))
        with Converter(
                reader, output_directory, output_file=o_file, frame=frame,
                row_limits=row_lims, col_limits=col_lims, output_format=output_format,
                check_older_version=check_older_version) as converter:
            converter.write_data(max_block_size=max_block_size)
