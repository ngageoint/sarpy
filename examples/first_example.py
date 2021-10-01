"""
Basic First Example
===================

This indicates the basic functionality for reading a complex format data set. It
is intended that this script will be read, and the relevant portions run
individually.


Learning Python
----------------

It seems somewhat common for some users of sarpy to be experts in SAR, but not
well versed with Python. If this describes you (and even if it doesn't), the
built-in `help()` function can often help clear up confusion found during an
interactive Python session.

Check out a given module, class, or function using it's path
>>> help('sarpy')
or
>>> help('sarpy.io.complex.sicd.SICDReader')

or, check out a variable that has been defined somehow
>>> from sarpy.io.complex.converter import open_complex
>>> reader = open_complex('<path to file>')
>>> help(reader)
"""


"""
General file opening
--------------------

Open a **complex format file** using a general purpose opener from a specified 
file name.

This general purpose opener iterates over the reader objects defined in
sarpy.io.complex, trying each and attempting to catch exceptions. It returns
the first one that works, or raises an exception if none of the readers work.
Note that this exception catching process, though we attempt to make everything
completely robust, can sometimes confusingly hide errors in the file identification,
parsing and/or interpretation.

It is important to note that this opener will not open any file that is not
**complex format** (like a WBID or detected image).
"""
from sarpy.io.complex.converter import open_complex


reader = open_complex('<path to file>')
# this will return an instance of one of the reader classes defined in the modules in sarpy.io.complex
print(type(reader))


"""
Direct file opening
-------------------

If you know the file type of your file, you may want to use the correct reader
class directly, especially if the general opener is not working.
"""
from sarpy.io.complex.sicd import SICDReader
reader = SICDReader('<path to file>')  # same class as referenced above


"""
SICD metadata structure
-----------------------

Access the SICD structure or tuple of structures associated with the reader.
Note that the sicd structure is defined using elements of sarpy.io.complex.sicd_elements

A sicd file will necessarily be composed of a single image, but other file formats
(like sentinel or radarsat) often contain multiple images combined into a single
package (i.e. multiple polarizations or other aggregate collections).
"""

# nebulous contents - this will be an instance of the sicd structure, or a tuple
# of sicd structures
nebulous_contents = reader.sicd_meta

# Unified access - this will always be a tuple of sicd structures,
# with one sicd structure per image
sicd_tuple = reader.get_sicds_as_tuple()

the_sicd = sicd_tuple[0]  # access the desired sicd structure
# provide a human readable, if long, contents to terminal
print(the_sicd)
# get xml string representation
xml_string = the_sicd.to_xml_string(tag='SICD')
# get json friendly dict representation
dict_representation = the_sicd.to_dict()

# access field values
print(the_sicd.CollectionInfo.CollectorName)


"""
Read complex pixel data
-----------------------

The recommended methodology uses slice notation. The basic syntax is as:
>>> data = reader[row_start:row_end:row_step, col_start:col_end:col_step, image_index=0]
"""
# in the event of a single image segment
all_data = reader[:]  # or reader[:, :] - reads all data
decimated_data = reader[::10, ::10] # reads every 10th pixel

# in the event of multiple image segments, use another slice index like
all_data = reader[:, :, 0]

"""
Opening remote file
-------------------

The SICD reader (and also SIDD reader) have been implemented to accept binary file-like 
objects, specifically intended to enable 

**Speed/efficiency:** Files read using the file system (i.e. via file name
or local file-like object) are read efficiently via numpy memory map. Reading
across a network file system, commonly encountered as reading from a file-share
drive mounted to your local system, maintains the efficiency of numpy memory map
usage, but the speed will be impacted (perhaps significantly) by network latency.

It should be noted that the flexibility of reading using a file-like object comes
at a significant efficiency and speed cost, particularly for reading decimated or
down-selected data. A numpy memory map can not be utilized (at least as of May 2021)
for a non-file object, and reading/interpreting data becomes a fully manual
and non-optimized process. The entire continguous chunk of data containing the
desired segment of data will be read, then down-selected. This is to accommodate
for the overhead of the connection request for remote reading - simple bench marks
indicate that the bottleneck for performing a remote read is clearly the connection
request, and presents no good opportunity for clear optimization.
"""

# for the purposes of general purpose example, we reference a basic example sicd
# file hosted for the SIX project usage. It is recommended to use local files, as
# described below
import smart_open
file_object = smart_open.open(
    'https://six-library.s3.amazonaws.com/sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments.nitf',
    mode='rb',  # must be opened in binary mode
    buffering=4*1024*1024)  # it has been observed that setting a manual buffer size may help



"""
Basic data plot
---------------

Show some basic plots of the data.
 
**Note:** the sarpy_apps project provides robust interactive tools.
"""

from matplotlib import pyplot
from sarpy.visualization.remap import Density

remap_function = Density()

fig, axs = pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5))
axs.imshow(remap_function(all_data), cmap='gray')
pyplot.show()


"""
Convert to SICD format
----------------------

Convert a complex dataset, in any format handled by sarpy, to SICD
"""

from sarpy.io.complex.converter import conversion_utility
conversion_utility('<complex format file>', '<output_directory>')
