"""This package contains a framework for handling SAR complex image data.

Example of how to use this package:
    import complex  # This package
    reader_object = complex.open(some_file_name)  # Determines the file format
                                                  # and returns an object that
                                                  # can read that format
    print(reader_object.sicdmeta)  # Displays metadata from file in SICD format
    reader_object.sicdmeta  # Displays the XML representation
    # read_chip method of reader object has the following syntax:
    # read_chip[start:stop:step, start:stop:step]
    # Examples:
    cdata = reader_object.read_chip[0:1000:2,0:1000:2]
    # Reads every other row and column from the first thousand rows and columns.
    cdata = reader_object.read_chip[...] # Reads all complex data from file

    # Convert SAR complex data from a format recognized by the readers into a SICD file
    complex.convert(input_file_name, sicd_output_file_name, output_format='SICD')

"""
# TODO: HIGH - update this docstring to include well-formatted example code
#   The structure of the module is confusing. We should probably move the code down, and just have imports here.

import math
import os
import pkgutil
import sys


__classification__ = "UNCLASSIFIED"


class Reader(object):  # TODO: HIGH - update this to sensibility. file type object? Move down.
    """Currently a placeholder used as an abstract class."""
    # Any subclass deriving from this class should implement these things:
    sicdmeta = None

    def read_chip(self, dim1range, dim2range):
        pass


class Writer(object):  # TODO: HIGH - update this to sensibility. file-type object? Move down.
    """Currently a placeholder used as an abstract class."""
    # Any subclass deriving from this class should implement these things:
    filename = None
    sicdmeta = None

    def write_chip(self, data, start_indices):
        pass


# TODO: HIGH - it is generally a bad idea to override a built-in name like this.
#   Can we refactor this name? Get clarification with Wade.
def open(filename):
    """
    Determines format of complex SAR data, and returns appropriate Reader class instance.
    :param filename:
    :return: Reader class instance
    :raises: IOError
    """

    # Note: open is also a built-in Python function.
    if not os.path.exists(filename) or not os.path.isfile(filename):
        raise IOError('File {} does not exist.'.format(filename))
    if not os.access(filename, os.R_OK):
        raise IOError('File {} does exists, but is not readable.'.format(filename))

    # Get a list of all the modules that describe SAR complex image file
    # formats.  For this to work, all of the file format handling modules
    # must be in the top level of this package.
    # TODO: MEDIUM - this is unnecessarily confusing. We should do this explicitly.
    mod_iter = pkgutil.iter_modules(__path__, __name__ + '.')
    module_names = [name for loader, name, ispkg in mod_iter if not ispkg]
    modules = [sys.modules[names] for names in module_names if __import__(names)]
    # Determine file format and return the proper file reader object
    for current_mod in modules:
        if hasattr(current_mod, 'isa'):  # Make sure its a file format handling module
            try:
                file_obj = current_mod.isa(filename)  # isa returns None if format does not match
                if file_obj:
                    return file_obj(filename)  # return file reader object of correct type
            except Exception:
                pass
    raise IOError('Unable to determine complex image format.')


def convert(input_filename,
            output_filename,
            frames=None,
            output_format='SICD',
            row_limits=None,
            column_limits=None,  # [start, stop], zero-based
            max_block_size=2**26):  # Default block size is roughly 50 MB
    """
    Converts one SAR complex data file format to another.
    :param input_filename: path of file to be converted
    :param output_filename: path of file to be created
    :param frames: None, int, or list of ints specifying which frame(s) to convert. If None, then all frames.
    :param output_format: The output file format to write.
    :param row_limits: None (all) or [start, stop]. Follows Python convention range of exclusivity.
    :param column_limits: None (all) or [start, stop]. Follows Python convention range of exclusivity.
    :param max_block_size: element block size used for the copy
    :raises: IOError
    """

    ro = open(input_filename)
    # Allow for single frame or multiple frame files
    try:
        # TODO: MEDIUM - this is bad practice
        len(ro.sicdmeta)
        sicdmeta = ro.sicdmeta
        read_chip = ro.read_chip
    except (TypeError, AttributeError):
        sicdmeta = [ro.sicdmeta]
        read_chip = [ro.read_chip]

    try:
        # TODO: MEDIUM - this is bad practice
        len(frames)
    except (TypeError, AttributeError):
        if frames is None:  # Default to all frames
            frames = range(len(read_chip))
        else:  # Scalar for single frame passed in
            frames = [frames]

    # Get a list of all the modules that describe SAR complex image file
    # formats.  For this to work, all of the file format handling modules
    # must be in the top level of this package.
    # TODO: MEDIUM - again this is unnecessarily complicated. It also should not be mysterious whether we
    #   can write a specified format. We should clearly have a map for such items here.
    module_names = [name for loader, name, ispkg
                    in pkgutil.iter_modules(__path__, __name__ + '.')
                    if (not ispkg and __import__(name))]
    # Find the proper module for writing the requested format and write file
    for name in module_names:
        if (hasattr(sys.modules[name], 'Writer') and  # Make sure its a module that can write
           name.rsplit('.', 1)[1].upper() == output_format.upper()):  # Requested format
            # Iterate through potentially multiple datasets within a single file
            for i in frames:
                if len(frames) > 1:
                    # Unique filename for each frame
                    path, fname = os.path.split(output_filename)
                    fname, ext = os.path.splitext(fname)
                    new_output_filename = os.path.join(path, ('%s%.3d' % (fname, i)) +
                                                       ext)
                else:
                    new_output_filename = output_filename

                # File chipping
                if column_limits is None:
                    column_limits = [0, sicdmeta[i].ImageData.NumCols]
                if row_limits is None:
                    row_limits = [0, sicdmeta[i].ImageData.NumRows]
                if (column_limits != [0, sicdmeta[i].ImageData.NumCols] or
                   row_limits != [0, sicdmeta[i].ImageData.NumRows]):
                    sicdmeta[i].ImageData.NumRows = int(row_limits[1]) - int(row_limits[0])
                    sicdmeta[i].ImageData.NumCols = int(column_limits[1]) - int(column_limits[0])
                    sicdmeta[i].ImageData.FirstRow = int(sicdmeta[i].ImageData.FirstRow +
                                                         int(row_limits[0]))
                    sicdmeta[i].ImageData.FirstCol = int(sicdmeta[i].ImageData.FirstCol +
                                                         int(column_limits[0]))
                    from .sicd import update_corners  # Can't do this at top of module  # TODO: HIGH - unbreak the structure
                    update_corners(sicdmeta[i])

                wo = sys.modules[name].Writer(new_output_filename, sicdmeta[i])
                # Progressively copy data in blocks, in case data is too large to
                # hold in memory.  Each block is a set of consecutive rows.
                # bytes_per_row assumes max of 8-byte elements (largest allowed in SICD)
                bytes_per_row = sicdmeta[i].ImageData.NumCols*8
                rows_per_block = max(1,  # If single row is larger than block size
                                     math.floor(float(max_block_size) /
                                                float(bytes_per_row)))
                num_blocks = int(max(1,  # Allows for max_block_size = inf
                                     math.ceil(float(sicdmeta[i].ImageData.NumRows) /
                                               float(rows_per_block))))
                for j in range(num_blocks):
                    block_start = j * int(rows_per_block) + int(row_limits[0])
                    block_end = int(min(block_start + rows_per_block, row_limits[1]))
                    data = read_chip[i]([block_start, block_end],
                                        [int(column_limits[0]), int(column_limits[1])])
                    wo.write_chip(data, (j * int(rows_per_block), 0))
                del wo
            return
    raise IOError('Unable to locate complex image writer of requested format.')
