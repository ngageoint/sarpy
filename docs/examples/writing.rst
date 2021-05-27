Format Conversion and File or Product Creation
==============================================

The basic methods for complex format conversion and derived product creation.


Convert to SICD format
----------------------

Convert a SICD type dataset (in any format handled by sarpy) to SICD format.

.. code-block:: python

    from sarpy.io.complex.converter import conversion_utility
    conversion_utility('<path to complex format file>', '<output_directory>')


This will create one SICD file in the output directory per image in the input file.
See the :meth:`sarpy.io.complex.converter.conversion_utility` documentation for
the an outline of the more robust capabilities for chipping, explicit output
filenames, etc.

This can also be accomplished using a command-line utility as

>>> python -m sarpy.utils.convert_to_sicd <input file> <output directory>

For a basic help, check

>>> python -m sarpy.utils.convert_to_sicd --help

Derived Product Creation from SICD
----------------------------------

There are basic ortho-rectification utilities in sarpy, and these are used in a
collection of methods to create data products derived from a SICD type file
following the SIDD standard.

.. code-block:: python

    import os

    from sarpy.io.complex.converter import open_complex
    from sarpy.processing.ortho_rectify import BivariateSplineMethod, \
        NearestNeighborMethod, PGProjection
    from sarpy.io.product.sidd_product_creation import create_detected_image_sidd, \
        create_csi_sidd, create_dynamic_image_sidd

    # open a sicd type file
    reader = open_complex('<sicd type object file name>')
    # create an orthorectification helper for specified sicd index
    ortho_helper = NearestNeighborMethod(reader, index=0)

    # create a sidd version 2 detected image for the whole file
    create_detected_image_sidd(ortho_helper, '<output directory>', block_size=10, version=2)

    # create a sidd version 2 color sub-aperture image for the whole file
    create_csi_sidd(ortho_helper, '<output directory>', dimension=0, version=2)

    # create a sidd version 2 dynamic image/sub-aperture stack for the whole file
    create_dynamic_image_sidd(ortho_helper, '<output directory>', dimension=0, version=2)

See module :mod:`sarpy.io.product.sidd_product_creation` for more specific details.

This can also be accomplished using a command-line utility as

>>> python -m sarpy.utils.create_product <input file> <output directory>

For a basic help, check

>>> python -m sarpy.utils.create_product --help

KMZ Product Creation from SICD
------------------------------

There are a few basic utilities for producing a kmz overlay from a SICD type file.

.. code-block:: python

    import os
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.product.kmz_product_creation import create_kmz_view

    test_root = '<root directory>'
    reader = open_complex(os.path.join(test_root, '<file name>'))
    create_kmz_view(
        reader, test_root,
        file_stem='View-<something descriptive>',
        pixel_limit=2048,
        inc_collection_wedge=True)


See module :mod:`sarpy.io.product.kmz_product_creation` for more specific details.

This can also be accomplished using a command-line utility as

>>> python -m sarpy.utils.create_kmz <input file> <output directory> -v

For a basic help on the command-line, check

>>> python -m sarpy.utils.create_kmz --help
