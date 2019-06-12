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

import math
import os
import pkgutil
import sys

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"
__email__ = "wschwartzkopf@integrity-apps.com"


# "Abstract" classes
# Note that as of Python 2.6, we could use the abc.py module for a more
# restrictive abstract base class.  However, since we want this to work on
# Python 2.5, we have avoided that.  The definition here is merely a
# description of the convention we will be using for our classes, not a
# way to enforce this convention.
class Reader():
    """Currently a placeholder used as an abstract class."""
    # Any subclass deriving from this class should implement these things:
    sicdmeta = None

    def read_chip(self, dim1range, dim2range):
        pass


class Writer():
    """Currently a placeholder used as an abstract class."""
    # Any subclass deriving from this class should implement these things:
    filename = None
    sicdmeta = None

    def write_chip(self, data, start_indices):
        pass


# Functions
def open(filename):  # Note: open is also a built-in Python function.
    """Function to check format of file and open the appropriate file object."""
    # Check that file exists.  We could let this get caught in the isa functions
    # below, but putting a check here results in an error message that is more
    # obvious for the user to understand.
    if not os.path.exists(filename) or not os.path.isfile(filename):
        raise IOError('File ' + filename + ' does not exist.')
    # Get a list of all the modules that describe SAR complex image file
    # formats.  For this to work, all of the file format handling modules
    # must be in the top level of this package.
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
    # If for loop completes, no matching file format was found.
    raise IOError('Unable to determine complex image format.')


def convert(input_filename, output_filename, frames=None, output_format='SICD',
            row_limits=None, column_limits=None,  # [start, stop], zero-based
            max_block_size=2**26):  # Default block size is roughly 50 MB
    """Copy SAR complex data to a file of the specified format.

    Parameters
    ----------
    input_filename : str
       Name of file to convert.  Can be of any type supported by the readers in this package.
    output_filename : str
       The name of the output file.
    frame : int or list of int, optional
       Set of each frame to convert.  Default is all.
    output_format : {'SICD', 'SIO'}, optional
       The output file format to write.  Default is SICD.
    row_limits : [int, int], optional
       Rows start/stop. Default is all [0 NumRows]
    column_limits : [int, int], optional
       Columns start/stop. Default is all [0 NumCols]
    """
    ro = open(input_filename)
    # Allow for single frame or multiple frame files
    try:
        len(ro.sicdmeta)
        sicdmeta = ro.sicdmeta
        read_chip = ro.read_chip
    except (TypeError, AttributeError):
        sicdmeta = [ro.sicdmeta]
        read_chip = [ro.read_chip]
    try:
        len(frames)
    except (TypeError, AttributeError):
        if frames is None:  # Default to all frames
            frames = range(len(read_chip))
        else:  # Scalar for single frame passed in
            frames = [frames]
    # Get a list of all the modules that describe SAR complex image file
    # formats.  For this to work, all of the file format handling modules
    # must be in the top level of this package.
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
                if(column_limits is None):
                    column_limits = [0, sicdmeta[i].ImageData.NumCols]
                if(row_limits is None):
                    row_limits = [0, sicdmeta[i].ImageData.NumRows]
                if (column_limits != [0, sicdmeta[i].ImageData.NumCols] or
                   row_limits != [0, sicdmeta[i].ImageData.NumRows]):
                    sicdmeta[i].ImageData.NumRows = int(row_limits[1]) - int(row_limits[0])
                    sicdmeta[i].ImageData.NumCols = int(column_limits[1]) - int(column_limits[0])
                    sicdmeta[i].ImageData.FirstRow = int(sicdmeta[i].ImageData.FirstRow +
                                                         int(row_limits[0]))
                    sicdmeta[i].ImageData.FirstCol = int(sicdmeta[i].ImageData.FirstCol +
                                                         int(column_limits[0]))
                    from .sicd import update_corners  # Can't do this at top of module
                    update_corners(sicdmeta[i])

                wo = sys.modules[name].Writer(new_output_filename, sicdmeta[i])
                # Progressively copy data in blocks, in case data is too large to
                # hold in memory.  Each block is a set of consecutive rows.
                # bytes_per_row assumes max of 8-byte elements (largest allowed in SICD)
                bytes_per_row = sicdmeta[i].ImageData.NumCols * 8
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
    # If for loop completes, no matching file format was found.
    raise IOError('Unable to locate complex image writer of requested format.')
