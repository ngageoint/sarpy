Image/Data Reading
==================

The basics for complex format data reading is presented in the first example.
This builds on that example, and presents more details for other reader types.

Reader Confusion
----------------

In sarpy, the readers for different image types are generally implemented in one
of the subpackages:

- :mod:`sarpy.io.complex` - for complex images directly analogous to SICD format
- :mod:`sarpy.io.product` - for standard image derived products like SIDD and WBID
- :mod:`sarpy.io.phase_history` - for CPHD images/data
- :mod:`sarpy.io.received` - for CRSD images/data

Each of these supackages contains an :meth:`open` function (aliased from the
`converter` module), which should open eligible files of ONLY their given type.
For example, the :meth:`sarpy.io.complex.open` function will open a SICD or SLC
products from Cosmo Skymed, RadarSat, Sentinel, etc, but **will not** open a
SIDD/WBID, general NITF file which isn't analogous to a SICD, CPHD, or CRSD file.

SICD-Type Readers
-----------------

As generally outlined in the first example, the complex format readers are all
shaped to be directly analogous for a SICD reader. The readers are all defined
in the :mod:`sarpy.io.complex` subpackage.

The general opening method is defined as :meth:`sarpy.io.complex.converter.open_complex`,
with basic usage as indicated by the below example.

.. code-block:: python

    from sarpy.io.complex.converter import open_complex

    reader = open_complex('<path to file>')

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}'.format(reader.data_size))
    print('image size as tuple = {}'.format(reader.get_data_size_as_tuple()))


Any reader returned by this function should be an extension of the class
:class:`sarpy.io.complex.base.SICDTypeReader`.

Some basic properties:

- We will have :code:`reader.reader_type = 'SICD'`, the image data
  will be of complex type, and the reader is for a format analogous to the SICD format.
- The image data sizes can be referenced using the :code:`reader.data_size` property
  (described here :attr:`sarpy.io.general.base.AbstractReader.data_size`) and/or the
  :code:`reader.get_data_size_as_tuple()` function
  (described here :meth:`sarpy.io.general.base.AbstractReader.get_data_size_as_tuple`).
- The image data can be read using slice notation
  :code:`data = reader[row_start:row_end:row_step, col_start:col_end:col_step, image_index=0]`.
  This data will have be recast or re-interpreted to be 64-bit complex data type,
  regardless of storage type.
- The SICD structures can be referenced using the :code:`reader.sicd_meta` property
  (described here :attr:`sarpy.io.complex.base.SICDTypeReader.sicd_meta`)
  and/or the :code:`reader.get_sicds_as_tuple()` function
  (described here :meth:`sarpy.io.complex.base.SICDTypeReader.get_sicds_as_tuple`).
- The image collection can be partitioned based on identical footprint, resolution,
  and collection frequency using the :code:`reader.get_sicd_partitions` method
  (described here :meth:`sarpy.io.complex.base.SICDTypeReader.get_sicd_partitions`).


Derived Product (SIDD-Type) Readers
-----------------------------------

Derived products, like WBID or SIDD files, have readers defined in the :mod:`sarpy.io.product`
subpackage. Such products are expected to be explicitly images derived from a SICD
type file, and processed to a standard (likely 8-bit) image for viewing/interpreting
by a human user.

The general opening method is defined as :meth:`sarpy.io.product.converter.open_product`,
with basic usage as indicated by

.. code-block:: python

    from sarpy.io.product.converter import open_product
    reader = open_product('< path to file>')

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}'.format(reader.data_size))
    print('image size as tuple = {}'.format(reader.get_data_size_as_tuple()))

Any reader retruned by this function should be an extension of the class
:class:`sarpy.io.product.base.SIDDTypeReader`.

Some basic properties:

- We will have :code:`reader.reader_type = 'SIDD'`,
  the image data will be of 8 or 16 bit unsigned integer (monochromatic or RGB),
  and the reader is for a format analogous to the SIDD format.
- The image data sizes can be referenced using the :code:`reader.data_size` property
  (described here :attr:`sarpy.io.general.base.AbstractReader.data_size`) and/or the
  :code:`reader.get_data_size_as_tuple()` function
  (described here :meth:`sarpy.io.general.base.AbstractReader.get_data_size_as_tuple`).
- The image data can be read using slice notation
  :code:`data = reader[row_start:row_end:row_step, col_start:col_end:col_step, image_index=0]`
- The SIDD structures can be referenced as :code:`reader.sidd_meta` property (
  described here :attr:`sarpy.io.product.base.SIDDTypeReader.sidd_meta`).
- **If the SICD structure from which the product is derived is populated in the product file,**
  then the SICD structures can be referenced using :attr:`sarpy.io.product.base.SIDDTypeReader.sicd_meta`.


Phase History (CPHD) Readers
----------------------------

The Compensated Phase History Data (CPHD) have readers defined in the :mod:`sarpy.io.phase_history`
subpackage. The standard for CPHD version 0.3 is significantly different than
the standard for version 1.0, and separate readers for version 0.3
(:class:`sarpy.io.phase_history.cphd.CPHDReader0_3`) and for version 1.0
(:class:`sarpy.io.phase_history.cphd.CPHDReader1_0`) are implemented for each;
both of which extend the common abstract parent given in
:class:`sarpy.io.phase_history.cphd.CPHDReader`.

The general opening method is defined as :meth:`sarpy.io.phase_history.converter.open_phase_history`,
with basic usage as indicated by

.. code-block:: python

    from sarpy.io.phase_history.converter import open_phase_history
    reader = open_phase_history('< path to file>')

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}'.format(reader.data_size))
    print('image size as tuple = {}'.format(reader.get_data_size_as_tuple()))


Any reader returned by the function will be an extension of the class
:class:`sarpy.io.phase_history.base.CPHDTypeReader`.

Some basic properties:

- We will have :code:`reader.reader_type = 'CPHD'`, and the image data will
  be of complex type.
- The CPHD version can be accessed via the :code:`reader.cphd_version` property
  (see :attr:`sarpy.io.phase_history.CPHDReader.cphd_version`).
- The image data sizes can be referenced using the :code:`reader.data_size` property
  (described here :attr:`sarpy.io.general.base.AbstractReader.data_size`) and/or the
  :code:`reader.get_data_size_as_tuple()` function
  (described here :meth:`sarpy.io.general.base.AbstractReader.get_data_size_as_tuple`).
- The phase history (or image) data can be read using slice notation
  :code:`data = reader[row_start:row_end:row_step, col_start:col_end:col_step, image_index=0]`.
  This data will have be recast or re-interpreted to be 64-bit complex data type,
  regardless of storage type.
- The full Per Vector Parameter (PVP) collection for a given range can be read using
  the :code:`reader.read_pvp_array()` function
  (see :meth:`sarpy.io.phase_history.base.CPHDTypeReader.read_pvp_array`).
- A single PVP variable for a given range can be read using the :code:`reader.read_pvp_variable()`
  function (see :meth:`sarpy.io.phase_history.base.CPHDTypeReader.read_pvp_variable`).
- For CPHD Version 1.0, a support array can be read for the given range using
  the :code:`reader.read_support_array()` function
  (see :meth:`sarpy.io.phase_history.base.CPHDTypeReader.read_support_array`).


Received Signal Data (CRSD) Readers
-----------------------------------

The Compensated Received Signal Data (CRSD) have readers defined in the :mod:`sarpy.io.received`
subpackage. The general opening method is defined as :meth:`sarpy.io.received.converter.open_received`,
with basic usage as indicated by

.. code-block:: python

    from sarpy.io.received.converter import open_received
    reader = open_received('< path to file>')

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}'.format(reader.data_size))
    print('image size as tuple = {}'.format(reader.get_data_size_as_tuple()))


Any reader returned by the function will be an extension of the class
:class:`sarpy.io.received.base.CRSDTypeReader`.

Some basic properties:

- We will have :code:`reader.reader_type = 'CRSD'`, and the image data will
  be of complex type.
- The CRSD version can be accessed via the :code:`reader.crsd_version` property
  (see :attr:`sarpy.io.received.crsd.CRSDReader.crsd_version`).
- The image data sizes can be referenced using the :code:`reader.data_size` property
  (described here :attr:`sarpy.io.general.base.AbstractReader.data_size`) and/or the
  :code:`reader.get_data_size_as_tuple()` function
  (described here :meth:`sarpy.io.general.base.AbstractReader.get_data_size_as_tuple`).
- The received signal (or image) data can be read using slice notation
  :code:`data = reader[row_start:row_end:row_step, col_start:col_end:col_step, image_index=0]`.
  This data will have be recast or re-interpreted to be 64-bit complex data type,
  regardless of storage type.
- The full Per Vector Parameter (PVP) collection for a given range can be read using
  the :code:`reader.read_pvp_array()` function
  (see :meth:`sarpy.io.received.base.CRSDTypeReader.read_pvp_array`).
- A single PVP variable for a given range can be read using the :code:`reader.read_pvp_variable()`
  function (see :meth:`sarpy.io.received.base.CRSDTypeReader.read_pvp_variable`).
- For CPHD Version 1.0, a support array can be read for the given range using
  the :code:`reader.read_support_array()` function
  (see :meth:`sarpy.io.received.base.CRSDTypeReader.read_support_array`).


NITF Option of Last Resort
--------------------------

Some support for general NITF file (not SICD, SIDD, or some radar specific format)
opening provided in the :mod:`sarpy.io.general` subpackage. This is certainly not fully
fledged support for every type of NITF, particularly for reading compressed image data.

The commandline utility defined in :mod:`sarpy.utils.nitf_utils` may be very useful
for a variety of metadata extraction purposes.
