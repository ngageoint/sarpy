"""
Methods for creating a variety of SIDD products.

Examples
--------
Create a variety of sidd products.

.. code-block:: python

    import os

    from sarpy.io.complex.converter import open_complex
    from sarpy.processing.ortho_rectify import BivariateSplineMethod, NearestNeighborMethod, PGProjection
    from sarpy.io.product.sidd_product_creation import create_detected_image_sidd, create_csi_sidd, create_dynamic_image_sidd

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
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import os
from sarpy.processing.ortho_rectify import OrthorectificationHelper, \
    FullResolutionFetcher, OrthorectificationIterator
from sarpy.io.product.sidd_structure_creation import create_sidd_structure
from sarpy.processing.csi import CSICalculator
from sarpy.processing.subaperture import SubapertureCalculator, SubapertureOrthoIterator
from sarpy.io.product.sidd import SIDDWriter
from sarpy.io.general.base import SarpyIOError
from sarpy.visualization.remap import MonochromaticRemap, NRL

DEFAULT_IMG_REMAP = NRL
DEFAULT_CSI_REMAP = NRL
DEFAULT_DI_REMAP = NRL


def _validate_filename(output_directory, output_file, sidd_structure):
    """
    Validate the output filename.

    Parameters
    ----------
    output_directory : str
    output_file : None|str
    sidd_structure

    Returns
    -------
    str
    """

    if output_file is None:
        # noinspection PyProtectedMember
        fstem = os.path.split(sidd_structure.NITF['SUGGESTED_NAME']+'.nitf')[1]
    else:
        fstem = os.path.split(output_file)[1]

    full_filename = os.path.join(os.path.expanduser(output_directory), fstem)
    if os.path.exists(full_filename):
        raise SarpyIOError('File {} already exists.'.format(full_filename))
    return full_filename


def _validate_remap_function(remap_function):
    """
    Verify that the given monochromatic remap function is viable for SIDD
    production.

    Parameters
    ----------
    remap_function : MonochromaticRemap
    """

    if not isinstance(remap_function, MonochromaticRemap):
        raise TypeError('remap_function must be an instance of MonochromaticRemap')
    if remap_function.bit_depth not in [8, 16]:
        raise TypeError('remap_function usage for SIDD requires 8 or 16 bit output')


def create_detected_image_sidd(
        ortho_helper, output_directory, output_file=None, block_size=10, dimension=0,
        bounds=None, version=2, include_sicd=True, remap_function=None):
    """
    Create a SIDD version of a basic detected image from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    dimension : int
        Which dimension to split over in block processing? Must be either 0 or 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    version : int
        The SIDD version to use, must be one of 1 or 2.
    include_sicd : bool
        Include the SICD structure in the SIDD file?
    remap_function : None|MonochromaticRemap
        The applied remap function. If one is not provided, then a default is
        used. Required global parameters will be calculated if they are missing,
        so the internal state of this remap function may be modified.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        import os

        from sarpy.io.complex.converter import open_complex
        from sarpy.processing.ortho_rectify import BivariateSplineMethod, NearestNeighborMethod, PGProjection
        from sarpy.io.product.sidd_product_creation import create_detected_image_sidd

        reader = open_complex('<sicd type object file name>')
        ortho_helper = NearestNeighborMethod(reader, index=0)

        # create a sidd version 2 file for the whole file
        create_detected_image_sidd(ortho_helper, '<output directory>', block_size=10, version=2)
    """

    if not os.path.isdir(output_directory):
        raise SarpyIOError('output_directory {} does not exist or is not a directory'.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(
            'ortho_helper is required to be an instance of OrthorectificationHelper, '
            'got type {}'.format(type(ortho_helper)))

    if remap_function is None:
        remap_function = DEFAULT_IMG_REMAP(override_name='IMG_DEFAULT')
    _validate_remap_function(remap_function)

    # construct the ortho-rectification iterator - for a basic data fetcher
    calculator = FullResolutionFetcher(
        ortho_helper.reader, dimension=dimension, index=ortho_helper.index, block_size=block_size)
    ortho_iterator = OrthorectificationIterator(
        ortho_helper, calculator=calculator, bounds=bounds,
        remap_function=remap_function, recalc_remap_globals=False)

    # create the sidd structure
    ortho_bounds = ortho_iterator.ortho_bounds
    sidd_structure = create_sidd_structure(
        ortho_helper, ortho_bounds,
        product_class='Detected Image', pixel_type='MONO{}I'.format(remap_function.bit_depth), version=version)
    # set suggested name
    sidd_structure.NITF['SUGGESTED_NAME'] = ortho_helper.sicd.get_suggested_name(ortho_helper.index)+'_IMG'

    # create the sidd writer
    full_filename = _validate_filename(output_directory, output_file, sidd_structure)
    writer = SIDDWriter(full_filename, sidd_structure, ortho_helper.sicd if include_sicd else None)

    # iterate and write
    for data, start_indices in ortho_iterator:
        writer(data, start_indices=start_indices, index=0)


def create_csi_sidd(
        ortho_helper, output_directory, output_file=None, dimension=0,
        block_size=30, bounds=None, version=2, include_sicd=True, remap_function=None):
    """
    Create a SIDD version of a Color Sub-Aperture Image from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    dimension : int
        The dimension over which to split the sub-aperture.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    version : int
        The SIDD version to use, must be one of 1 or 2.
    include_sicd : bool
        Include the SICD structure in the SIDD file?
    remap_function : None|MonochromaticRemap
        The applied remap function. For csi processing, this must explicitly be
        an 8-bit remap. If one is not provided, then a default is used. Required
        global parameters will be calculated if they are missing, so the internal
        state of this remap function may be modified.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        import os
        from sarpy.io.complex.converter import open_complex
        from sarpy.io.product.sidd_product_creation import create_csi_sidd
        from sarpy.processing.csi import CSICalculator
        from sarpy.processing.ortho_rectify import NearestNeighborMethod

        reader = open_complex('<sicd type object file name>')
        ortho_helper = NearestNeighborMethod(reader, index=0)
        create_csi_sidd(ortho_helper, '<output directory>', dimension=0, version=2)

    """

    if not os.path.isdir(output_directory):
        raise SarpyIOError('output_directory {} does not exist or is not a directory'.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(
            'ortho_helper is required to be an instance of OrthorectificationHelper, '
            'got type {}'.format(type(ortho_helper)))

    # construct the CSI calculator class
    csi_calculator = CSICalculator(
        ortho_helper.reader, dimension=dimension, index=ortho_helper.index, block_size=block_size)

    if remap_function is None:
        remap_function = DEFAULT_CSI_REMAP(override_name='CSI_DEFAULT', bit_depth=8)
    _validate_remap_function(remap_function)
    if remap_function.bit_depth != 8:
        raise ValueError('The CSI SIDD specifically requires an 8-bit remap function.')

    # construct the ortho-rectification iterator
    ortho_iterator = OrthorectificationIterator(
        ortho_helper, calculator=csi_calculator, bounds=bounds,
        remap_function=remap_function, recalc_remap_globals=False)

    # create the sidd structure
    ortho_bounds = ortho_iterator.ortho_bounds
    sidd_structure = create_sidd_structure(
        ortho_helper, ortho_bounds,
        product_class='Color Subaperture Image', pixel_type='RGB24I', version=version)
    # set suggested name
    sidd_structure.NITF['SUGGESTED_NAME'] = csi_calculator.sicd.get_suggested_name(csi_calculator.index)+'_CSI'

    # create the sidd writer
    full_filename = _validate_filename(output_directory, output_file, sidd_structure)
    writer = SIDDWriter(full_filename, sidd_structure, csi_calculator.sicd if include_sicd else None)

    # iterate and write
    for data, start_indices in ortho_iterator:
        writer(data, start_indices=start_indices, index=0)


def create_dynamic_image_sidd(
        ortho_helper, output_directory, output_file=None, dimension=0, block_size=10,
        bounds=None, frame_count=9, aperture_fraction=0.2, method='FULL', version=2,
        include_sicd=True, remap_function=None):
    """
    Create a SIDD version of a Dynamic Image (Sub-Aperture Stack) from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    dimension : int
        The dimension over which to split the sub-aperture.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    frame_count : int
        The number of frames to calculate.
    aperture_fraction : float
        The relative size of each aperture window.
    method : str
        The subaperture processing method, which must be one of
        `('NORMAL', 'FULL', 'MINIMAL')`.
    version : int
        The SIDD version to use, must be one of 1 or 2.
    include_sicd : bool
        Include the SICD structure in the SIDD file?
    remap_function : None|MonochromaticRemap
        The applied remap function. If one is not provided, then a default is
        used. Required global parameters will be calculated if they are missing,
        so the internal state of this remap function may be modified.

    Returns
    -------
    None

    Examples
    --------
    Create a basic dynamic image.

    .. code-block:: python

        import os
        from sarpy.io.complex.converter import open_complex
        from sarpy.io.product.sidd_product_creation import create_dynamic_image_sidd
        from sarpy.processing.csi import CSICalculator
        from sarpy.processing.ortho_rectify import NearestNeighborMethod

        reader = open_complex('<sicd type object file name>')
        ortho_helper = NearestNeighborMethod(reader, index=0)
        create_dynamic_image_sidd(ortho_helper, '<output directory>', dimension=0, version=2)
    """

    if not os.path.isdir(output_directory):
        raise SarpyIOError('output_directory {} does not exist or is not a directory'.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(
            'ortho_helper is required to be an instance of OrthorectificationHelper, '
            'got type {}'.format(type(ortho_helper)))

    # construct the subaperture calculator class
    subap_calculator = SubapertureCalculator(
        ortho_helper.reader, dimension=dimension, index=ortho_helper.index, block_size=block_size,
        frame_count=frame_count, aperture_fraction=aperture_fraction, method=method)

    if remap_function is None:
        remap_function = DEFAULT_DI_REMAP(override_name='DI_DEFAULT')
    _validate_remap_function(remap_function)

    # construct the ortho-rectification iterator
    ortho_iterator = SubapertureOrthoIterator(
        ortho_helper, calculator=subap_calculator, bounds=bounds,
        remap_function=remap_function, recalc_remap_globals=False, depth_first=True)

    # create the sidd structure
    ortho_bounds = ortho_iterator.ortho_bounds
    sidd_structure = create_sidd_structure(
        ortho_helper, ortho_bounds,
        product_class='Dynamic Image', pixel_type='MONO{}I'.format(remap_function.bit_depth), version=version)
    # set suggested name
    sidd_structure.NITF['SUGGESTED_NAME'] = subap_calculator.sicd.get_suggested_name(subap_calculator.index)+'__DI'
    the_sidds = []
    for i in range(subap_calculator.frame_count):
        this_sidd = sidd_structure.copy()
        this_sidd.ProductCreation.ProductType = 'Frame {}'.format(i+1)
        the_sidds.append(this_sidd)

    # create the sidd writer
    if output_file is None:
        # noinspection PyProtectedMember
        full_filename = os.path.join(output_directory, sidd_structure.NITF['SUGGESTED_NAME']+'.nitf')
    else:
        full_filename = os.path.join(output_directory, output_file)
    if os.path.exists(os.path.expanduser(full_filename)):
        raise SarpyIOError('File {} already exists.'.format(full_filename))
    writer = SIDDWriter(full_filename, the_sidds, subap_calculator.sicd if include_sicd else None)

    # iterate and write
    for data, start_indices, the_frame in ortho_iterator:
        writer(data, start_indices=start_indices, index=the_frame)
