Basic First Example - Complex Image
===================================

This set of examples is centered on issues and basic functionality for a
**complex format** image, which is the most likely to be of general importance
to users.

Learning Python
---------------

It seems somewhat common for some users of sarpy to be experts in SAR, but not
well versed with Python. If this describes you (and even if it doesn't), the
built-in :meth:`help` function can often help clear up confusion found during an
interactive Python session.

Check out a given module, class, or function using it's path/module structure

>>> help('sarpy')

or

>>> help('sarpy.io.complex.sicd.SICDReader')

or, check out a variable that has been defined somehow

>>> from sarpy.io.complex.converter import open_complex
>>> reader = open_complex('<path to file>')
>>> help(reader)


General complex format file opening
-----------------------------------
Open a **complex format file** using a general purpose opener from a specified
file name.

This general purpose opener iterates over the reader objects defined in
`sarpy.io.complex`, trying each and attempting to catch exceptions. It returns
the first one that works, or raises an exception if none of the readers work.
Note that this exception catching process, though we attempt to make everything
completely robust, can sometimes confusingly hide errors in the file identification,
parsing and/or interpretation.

.. Note::

    It is important to note that this opener will not open any file that is not
    **complex format**, like a WBID or detected image.

.. code-block:: python

    from sarpy.io.complex.converter import open_complex

    reader = open_complex('<path to sicd file>')
    # this will return an instance of one of the reader classes defined in the modules in sarpy.io.complex

    # to see which class, just check directly
    print(type(reader))


Direct complex format file opening
----------------------------------

If you know the file type (SICD, Sentinel, RadarSat, etc) of your file, it is advisable
to use the correct reader class directly.

The first reason to do this is efficiency. Using the correct reader avoids all
unnecessary steps in attempting using other format readers, catching the associated
exceptions, and continuing until success.

The second reason to do or try this is when the general opener process is not working.
Remember, the general complex opener function simply iterates over the collection of
complex format readers, *catching most exceptions*, and returning the first reader
instance that works. In this case, the catching of exceptions can actually lead to
confusion, because the underlying problem is being suppressed.

.. code-block:: python

    from sarpy.io.complex.sicd import SICDReader
    reader = SICDReader('<path to sicd file>')


SICD metadata structure
-----------------------

Access the SICD structure or tuple of structures associated with the reader.
Note that the sicd structure is defined using elements of `sarpy.io.complex.sicd_elements`

A sicd file will necessarily be composed of a single image, but other file formats
(like sentinel or radarsat) often contain multiple images combined into a single
package (i.e. multiple polarizations or other aggregate collections).

.. code-block:: python

    from sarpy.io.complex.converter import open_complex

    reader = open_complex('<path to complex file>')

    # nebulous contents - this will be a sicd structure, or a tuple of sicd structures
    nebulous_contents = reader.sicd_meta

    # Unified access - this will always be a tuple of sicd structures,
    # with one sicd structure per image
    sicd_tuple = reader.get_sicds_as_tuple()

    the_sicd = sicd_tuple[0]  # access the desired sicd structure

    # provide a human readable, if long, dump of contents to terminal
    print(the_sicd)

    # get xml string representation
    xml_string = the_sicd.to_xml_string(tag='SICD')

    # get json friendly dict representation
    dict_representation = the_sicd.to_dict()

    # access field values
    print(the_sicd.CollectionInfo.CollectorName)


The SICD structure access details are implemented as in :meth:`sarpy.io.complex.base.SICDTypeReader.get_sicds_as_tuple`.
The behavior of the SICD structure methods are implmeneted as in :meth:`sarpy.io.complex.sicd_elements.SICD.SICDType.to_xml_string`.

Read complex pixel data
-----------------------

In the image file(s), complex format data is generally stored with real and
imaginary components of either 16-bit integer or 32-bit floating point type.
When reading complex format data, regardless of storage format, the returned data
will be cast or redefined to be 64-bit complex data type, which has 32-bit floating
point real and imaginary components.

The recommended methodology uses slice notation, with basic syntax as:

.. code-block:: python

    # ... assumes previously defined reader instance

    # overall syntax
    data = reader[row_start:row_end:row_step, col_start:col_end:col_step, image_index=0]

    # in the event of a single image, or read from first image
    all_data = reader[:]  # or reader[:, :] - reads all data from the first image
    decimated_data = reader[::10, ::10] # reads every 10th pixel from the first image

    # read all data from the 3rd image (requires that there is one).
    third_image_data = reader[:, :, 2]

This behavior is implemented in  :meth:`sarpy.io.general.base.AbstractReader.__getitem__`.

Basic data plot and remap
-------------------------

The `sarpy_apps` project provides robust interactive tools, but here is a basic data
plot for simple scripting purposes.

.. code-block:: python

    from matplotlib import pyplot
    from sarpy.visualization.remap import Density

    # ... assumes previously defined reader instance

    remap_function = Density()
    # show the initial 500 x 500 chip, using the "standard" remap
    chip = reader[:500, :500]

    fig, axs = pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5))
    axs.imshow(remap_function(chip), cmap='gray')
    pyplot.show()


Opening remote file
-------------------

The SICD reader (and also SIDD reader) have been implemented to accept binary
file-like objects, specifically intended to enable remote reading from a given
url or S3 bucket.

Files read using the file system (i.e. via file name or local file-like object)
are read efficiently via numpy memory map. Reading across a network file system,
commonly encountered as reading from a file-share drive mounted to your local
system, maintains the efficiency of numpy memory map usage, but the speed will be
impacted (perhaps significantly) by network latency.

**Speed/efficiency Impact:** It should be noted that the flexibility of reading
using a file-like object comes at a significant efficiency and speed cost,
particularly for reading decimated or down-selected data. A numpy memory map can
not be utilized (at least as of May 2021) for a non-file object, and reading/interpreting
data becomes a fully manual and non-optimized process. The entire continguous chunk
of data containing the desired segment of data will be read, then down-selected.
This is to accommodate for the overhead of the connection request for remote
reading - simple bench marks indicate that the bottleneck for performing a remote
read is clearly the connection request, and presents no good opportunity for clear
optimization.

.. code-block:: python

    from sarpy.io.complex.sicd import SICDReader

    # for the purposes of general purpose example, we reference a basic example sicd
    # file hosted for the SIX project usage. It is recommended to use local files, as
    # described below
    import smart_open
    file_object = smart_open.open(
        'https://six-library.s3.amazonaws.com/sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments.nitf',
        mode='rb',  # must be opened in binary mode
        buffering=4*1024*1024)  # it has been observed that setting a manual buffer size may help

    reader = SICDReader(file_object)
    # this works as any reader object, with the caveats in reading efficiency as outlined above
