# -*- coding: utf-8 -*-

import os
import sys
import pkgutil
import numpy

from .base import BaseReader
from .sicd import SICDWriter
from .sio import SIOWriter
from .sicd_elements.SICD import SICDType

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
    sarpy.io.complex.base.BaseReader

    Raises
    ------
    IOError
    """

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
    output_file : str
       The name of the output file.
    frames : None|int|list
       Set of each frame to convert. Default is all.
    output_format : str
       The output file format to write, from {'SICD', 'SIO'}, optional.  Default is SICD.
    row_limits : None|int|Tuple[int, int]
       Rows start/stop. Default is all.
    column_limits : None|int|Tuple[int, int]
       Columns start/stop. Default is all.

    Returns
    -------
    None
    """

    def get_out_file_name(index):
        if not mangle_name:
            return output_file
        else:
            digits = int(numpy.ceil(numpy.log10(len(sicds))))
            frm_str = '{0:s}-1{:0' + str(digits) + 'd}{2:s}'
            fstem, fext = os.path.splitext(output_file)
            return frm_str.format(fstem, index, fext)

    if isinstance(frames, int):
        frames = (frames, )
    if not isinstance(frames, tuple):
        frames = tuple(frames)
    if len(frames) == 0:
        raise ValueError('The list of frames is empty.')

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

    if isinstance(sicds, SICDType):
        sicds = [sicds, ]
        frames = (0, )

    mangle_name = (len(sicds) > 1)

    for frame in frames:
        frame = int(frame)
        if not (0 <= frame < len(sicds)):
            raise ValueError(
                'Got a frame entry {}, but it must be between 0 and {}'.format(frame, len(sicds)))
        print('Converting frame {}'.format(frame))

        this_sicd = sicds[frame].copy()
        # TODO: update this_sicd to reflect data from row_limits and column_limits
        with writer_type(get_out_file_name(frame), this_sicd) as writer:
            # TODO: write it out
            data = ''
            writer.write_chip(data, start_indices='')
