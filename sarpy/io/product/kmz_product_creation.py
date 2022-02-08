"""
This module provides tools for creating kmz products for a SICD type element.

.. Note::
    Creation of ground overlays (i.e. image overlay) requires the optional
    Pillow dependency for image manipulation.

Examples
--------
Create a kmz overview for the contents of a sicd type reader.

.. code-block:: python

    import os
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.product.kmz_product_creation import create_kmz_view

    test_root = '<root directory>'
    reader = open_complex(os.path.join(test_root, '<file name>>'))
    create_kmz_view(reader, test_root,
                    file_stem='View-<something descriptive>',
                    pixel_limit=2048,
                    inc_collection_wedge=True)
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from typing import Union
import json
import os

import numpy

from sarpy.processing.rational_polynomial import SarpyRatPolyError
from sarpy.processing.ortho_rectify.base import FullResolutionFetcher, OrthorectificationIterator
from sarpy.processing.ortho_rectify.ortho_methods import OrthorectificationHelper, NearestNeighborMethod
from sarpy.processing.ortho_rectify.projection_helper import PGProjection, PGRatPolyProjection
from sarpy.io.kml import Document
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.utils import sicd_reader_iterator
from sarpy.geometry.geocoords import ecf_to_geodetic
from sarpy.visualization.remap import RemapFunction, NRL


try:
    # noinspection PyPackageRequirements
    import PIL
    import PIL.Image
except ImportError:
    PIL = None

logger = logging.getLogger(__name__)


def _create_sicd_styles(kmz_document):
    """
    Creates the appropriate styles for SICD usage.

    Parameters
    ----------
    kmz_document : Document

    Returns
    -------
    None
    """

    # bounding box style - maybe polygon, maybe corner points, clamped to ground
    label = {'color': 'ffc0c0c0', 'scale': '1.0'}
    icon = {'scale': '1.5', 'icon_ref': 'http://maps.google.com/mapfiles/kml/pushpin/blue-pushpin.png'}
    line = {'color': 'ccff5050', 'width': '2.0'}
    poly = {'color': '30ff5050'}
    kmz_document.add_style('bounding_high', label_style=label, icon_style=icon, line_style=line, poly_style=poly)
    label['scale'] = '0.75'
    icon['scale'] = '1.0'
    line['width'] = '1.0'
    kmz_document.add_style('bounding_low', label_style=label, icon_style=icon, line_style=line, poly_style=poly)
    kmz_document.add_style_map('bounding', 'bounding_high', 'bounding_low')

    # valid data style - basic polygon, probably clamped to ground
    line = {'color': 'cc5050ff', 'width': '2.0'}
    poly = {'color': '305050ff'}
    kmz_document.add_style('valid_high', line_style=line, poly_style=poly)
    line['width'] = '1.0'
    kmz_document.add_style('valid_low', line_style=line, poly_style=poly)
    kmz_document.add_style_map('valid', 'valid_high', 'valid_low')

    # scp - intended for basic point clamped to ground
    label = {'color': 'ff50c0c0', 'scale': '1.0'}
    icon = {'color': 'ff5050c0', 'scale': '1.5',
            'icon_ref': 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'}
    kmz_document.add_style('scp_high', label_style=label, icon_style=icon)
    label['scale'] = '0.75'
    icon['scale'] = '1.0'
    kmz_document.add_style('scp_low', label_style=label, icon_style=icon)
    kmz_document.add_style_map('scp', 'scp_high', 'scp_low')

    # arp position style - intended for gx track
    line = {'color': 'ff50ff50', 'width': '1.5'}
    label = {'color': 'ffc0c0c0', 'scale': '1.5'}
    icon = {'scale': '2.0', 'icon_ref': 'http://maps.google.com/mapfiles/kml/shapes/track.png'}
    poly = {'color': 'a050ff50'}
    kmz_document.add_style('arp_high', line_style=line, label_style=label, icon_style=icon, poly_style=poly)
    line['width'] = '1.0'
    label['scale'] = '1.0'
    icon['scale'] = '1.0'
    poly = {'color': '7050ff50'}
    kmz_document.add_style('arp_low', line_style=line, label_style=label, icon_style=icon, poly_style=poly)
    kmz_document.add_style_map('arp', 'arp_high', 'arp_low')

    # collection wedge style - intended as polygon
    line = {'color': 'ffa0a050', 'width': '1.5'}
    poly = {'color': 'a0a0a050'}
    kmz_document.add_style('collection_high', line_style=line, poly_style=poly)
    line['width'] = '1.0'
    poly = {'color': '70a0a050'}
    kmz_document.add_style('collection_low', line_style=line, poly_style=poly)
    kmz_document.add_style_map('collection', 'collection_high', 'collection_low')


def _get_sicd_name(sicd):
    """
    Gets the kml-styled name for the provided SICD.

    Parameters
    ----------
    sicd : SICDType

    Returns
    -------
    str
    """

    return sicd.CollectionInfo.CoreName


def _get_sicd_description(sicd):
    """
    Gets the kml-styled description for the provided SICD.

    Parameters
    ----------
    sicd : SICDType

    Returns
    -------
    str
    """

    o_sicd = sicd.copy()
    # junk the WgtFunct, it's huge and probably not interesting
    try:
        o_sicd.Grid.Row.WgtFunct = None
        o_sicd.Grid.Col.WgtFunct = None
    except AttributeError:
        pass

    return json.dumps(o_sicd.to_dict(), indent=1)


def _get_orthoiterator_description(ortho_iterator):
    """
    Get a description for the ortho_iterator details.

    Parameters
    ----------
    ortho_iterator : OrthorectificationIterator

    Returns
    -------
    str
    """

    return 'ortho-rectified image for {2:s}<br>' \
           'row resolution - {0:0.2f} meters<br>' \
           'column resolution - {1:0.2f} meters<br>' \
           'remap function - {3:s}'.format(
        ortho_iterator.ortho_helper.proj_helper.row_spacing,
        ortho_iterator.ortho_helper.proj_helper.col_spacing,
        _get_sicd_name(ortho_iterator.sicd),
        ortho_iterator.remap_function.name)


def _get_sicd_time_args(sicd, subdivisions=24):
    # type: (SICDType, Union[int, None]) -> (dict, Union[None, numpy.ndarray])
    """
    Fetch the SICD time arguments and array.

    Parameters
    ----------
    sicd : SICDType
    subdivisions : int|None

    Returns
    -------
    (dict, None|numpy.ndarray)
    """

    if sicd.Timeline is None or sicd.Timeline.CollectStart is None:
        return {}, None

    beg_time = sicd.Timeline.CollectStart.astype('datetime64[us]')
    if sicd.Timeline.CollectDuration is None:
        return {'when': str(beg_time)+'Z',}, None

    end_time = beg_time + int(sicd.Timeline.CollectDuration*1e6)
    if not isinstance(subdivisions, int) or subdivisions < 2:
        time_array = None
    else:
        time_array = numpy.linspace(0, sicd.Timeline.CollectDuration, subdivisions)
    return {'beginTime': str(beg_time)+'Z', 'endTime': str(end_time)+'Z'}, time_array


def _write_image_corners(kmz_document, sicd, time_args, folder, write_points=True):
    """
    Write the image corner.

    Parameters
    ----------
    kmz_document : Document
    sicd : SICDType
    time_args : dict
    folder : minidom.Element
    write_points : bool
        Write points, or a polygon?

    Returns
    -------
    None
    """

    if sicd.GeoData is None or sicd.GeoData.ImageCorners is None:
        return

    frm = '{1:0.8f},{0:0.8f},0'
    corners = sicd.GeoData.ImageCorners.get_array(dtype='float64')

    if numpy.any(~numpy.isfinite(corners)):
        logger.error('There are nonsense entries (nan or +/- infinity) in the corner locations array.')

    if write_points:
        names = ['FRFC', 'FRLC', 'LRLC', 'LRFC']
        for nam, corner in zip(names, corners):
            if numpy.any(~numpy.isfinite(corner)):
                continue
            coords = frm.format(*corner)
            placemark = kmz_document.add_container(par=folder, description='{} for {}'.format(nam, _get_sicd_name(sicd)),
                                                   styleUrl='#bounding')
            kmz_document.add_point(coords, par=placemark, altitudeMode='clampToGround', **time_args)
    else:
        # write the polygon
        coords = ' '.join(frm.format(*el) for el in corners if not numpy.any(~numpy.isfinite(el)))
        placemark = kmz_document.add_container(par=folder, description='image corners for {}'.format(_get_sicd_name(sicd)), styleUrl='#bounding')
        kmz_document.add_polygon(coords, par=placemark, altitudeMode='clampToGround', **time_args)


def _write_valid_area(kmz_document, sicd, time_args, folder):
    """
    Write the valid area polygon.

    Parameters
    ----------
    kmz_document : Document
    sicd : SICDType
    time_args : dict
    folder : minidom.Element

    Returns
    -------
    None
    """

    if sicd.GeoData is None or sicd.GeoData.ValidData is None:
        return

    frm = '{1:0.8f},{0:0.8f},0'
    valid_array = sicd.GeoData.ValidData.get_array(dtype='float64')
    if numpy.any(~numpy.isfinite(valid_array)):
        logger.error('There are nonsense entries (nan or +/- infinity) in the valid array location.')

    coords = ' '.join(frm.format(*el) for el in valid_array)
    coords += ' ' + frm.format(*valid_array[0, :])
    placemark = kmz_document.add_container(par=folder, description='valid data for {}'.format(_get_sicd_name(sicd)), styleUrl='#valid')
    kmz_document.add_polygon(coords, par=placemark, altitudeMode='clampToGround', **time_args)


def _write_scp(kmz_document, sicd, time_args, folder):
    """
    Write the csp location.

    Parameters
    ----------
    kmz_document : Document
    sicd : SICDType
    time_args : dict
    folder : minidom.Element

    Returns
    -------
    None
    """
    if sicd.GeoData is None or sicd.GeoData.SCP is None:
        return

    scp_llh = sicd.GeoData.SCP.LLH.get_array()
    if numpy.any(~numpy.isfinite(scp_llh)):
        logger.error('There are nonsense entries (nan or +/- infinity) in the scp location.')

    frm = '{1:0.8f},{0:0.8f},0'
    coords = frm.format(*scp_llh)
    placemark = kmz_document.add_container(par=folder, description='SCP for {}'.format(_get_sicd_name(sicd)), styleUrl='#scp')
    kmz_document.add_point(coords, par=placemark, altitudeMode='clampToGround', **time_args)


def _write_arp_location(kmz_document, sicd, time_args, time_array, folder):
    """

    Parameters
    ----------
    kmz_document : Document
    sicd : SICDType
    time_args : dict
    time_array : None|numpy.ndarray
    folder : minidom.Element

    Returns
    -------
    None|Numpy.ndarray
    """

    if time_array is None:
        return None

    if sicd.Position is not None and sicd.Position.ARPPoly is not None:
        arp_pos = sicd.Position.ARPPoly(time_array)
    elif sicd.SCPCOA.ARPPos is not None and sicd.SCPCOA.ARPVel is not None:
        arp_pos = sicd.SCPCOA.ARPPos.get_array() + numpy.outer(time_array, sicd.SCPCOA.ARPVel.get_array())
    else:
        return None

    arp_llh = ecf_to_geodetic(arp_pos)
    if numpy.any(~numpy.isfinite(arp_llh)):
        logger.error('There are nonsense entries (nan or +/- infinity) in the aperture location.')
    coords = ['{1:0.8f},{0:0.8f},{2:0.2f}'.format(*el) for el in arp_llh]
    whens = [str(sicd.Timeline.CollectStart.astype('datetime64[us]') + int(el*1e6)) + 'Z' for el in time_array]
    placemark = kmz_document.add_container(par=folder, description='aperture position for {}'.format(_get_sicd_name(sicd)), styleUrl='#arp', **time_args)
    kmz_document.add_gx_track(coords, whens, par=placemark, extrude=True, tesselate=True, altitudeMode='absolute')
    return arp_llh


def _write_collection_wedge(kmz_document, sicd, time_args, arp_llh, time_array, folder):
    """
    Writes the collection wedge.

    Parameters
    ----------
    kmz_document : Document
    sicd : SICDType
    time_args : dict
    arp_llh : None|numpy.ndarray
    time_array : None|numpy.ndarray
    folder : minidom.Element

    Returns
    -------
    None
    """

    if time_array is None or arp_llh is None:
        return

    if sicd.Position is not None and sicd.Position.GRPPoly is not None:
        grp = sicd.Position.GRPPoly(time_array)
    elif sicd.GeoData is not None and sicd.GeoData.SCP is not None:
        grp = numpy.reshape(sicd.GeoData.SCP.ECF.get_array(), (1, 3))
    else:
        return
    frm = '{1:0.8f},{0:0.8f},{2:0.2f}'
    grp_llh = ecf_to_geodetic(grp)

    if numpy.any(~numpy.isfinite(grp_llh)):
        logger.error('There are nonsense entries (nan or +/- infinity) in the scp/ground range locations.')

    coord_array = [frm.format(*el) for el in arp_llh]
    if len(grp_llh) > 1:
        coord_array.extend(frm.format(*el) for el in grp_llh[::-1, :])
    else:
        coord_array.append(frm.format(*grp_llh[0, :]))
    coord_array.append(frm.format(*arp_llh[0, :]))
    coords = ' '.join(coord_array)
    placemark = kmz_document.add_container(par=folder, description='collection wedge for {}'.format(_get_sicd_name(sicd)), styleUrl='#collection', **time_args)
    kmz_document.add_polygon(coords, par=placemark, extrude=False, tesselate=False, altitudeMode='absolute')


def _write_sicd_overlay(ortho_iterator, kmz_document, folder):
    """
    Write the orthorectified SICD ground overlay.

    Parameters
    ----------
    ortho_iterator : OrthorectificationIterator
    kmz_document : Document
    folder : minidom.Element

    Returns
    -------
    None
    """

    def reorder_corners(llh_in):
        return llh_in[::-1, :]

    if PIL is None:
        logger.error(
            'This functionality for writing kmz ground overlays requires the optional Pillow dependency.')
        return

    time_args, _ = _get_sicd_time_args(ortho_iterator.sicd, subdivisions=None)

    # create the output workspace
    if ortho_iterator.remap_function.bit_depth != 8:
        raise ValueError('The bit depth for the remap function must be 8, for now.')
    image_data = numpy.zeros(ortho_iterator.ortho_data_size, dtype=ortho_iterator.remap_function.output_dtype)
    # populate by iterating
    for data, start_indices in ortho_iterator:
        image_data[start_indices[0]:start_indices[0]+data.shape[0],
                   start_indices[1]:start_indices[1]+data.shape[1]] = data
    # create regionated overlay
    # convert image array to PIL image.
    img = PIL.Image.fromarray(image_data)  # this is to counteract the PIL treatment
    lat_lon_quad = reorder_corners(ortho_iterator.get_llh_image_corners())
    kmz_document.add_regionated_ground_overlay(
        img, folder, lat_lon_quad=lat_lon_quad[:, :2], img_format='JPEG',
        name='image overlay for {}'.format(_get_sicd_name(ortho_iterator.sicd)),
        description=_get_orthoiterator_description(ortho_iterator))


def prepare_kmz_file(file_name, **args):
    """
    Prepare a kmz document and archive for exporting.

    Parameters
    ----------
    file_name : str
    args
        Passed through to the Document constructor.

    Returns
    -------
    Document
    """

    document = Document(file_name=file_name, **args)
    _create_sicd_styles(document)
    return document


def add_sicd_geometry_elements(sicd, kmz_document, folder,
        inc_image_corners=True, inc_valid_data=False,
        inc_scp=False, inc_collection_wedge=True):
    """
    Write the geometry elements of a SICD.

    Parameters
    ----------
    sicd : SICDType
    kmz_document : Document
    folder : minidom.Element
    inc_image_corners : bool
        Include the image corners, if possible?
    inc_valid_data : bool
        Include the valid image area, if possible?
    inc_scp : bool
        Include the scp?
    inc_collection_wedge : bool
        Include the aperture location and collection wedge?

    Returns
    -------
    None
    """

    # let's define the time data for the SICD
    time_args, time_array = _get_sicd_time_args(sicd)

    # add the image corners/bounding box
    if inc_image_corners:
        _write_image_corners(kmz_document, sicd, time_args, folder)
    # add the valid data
    if inc_valid_data:
        _write_valid_area(kmz_document, sicd, time_args, folder)
    # write scp data
    if inc_scp:
        _write_scp(kmz_document, sicd, time_args, folder)
    # write arp position and collection wedge
    if inc_collection_wedge:
        arp_llh = _write_arp_location(kmz_document, sicd, time_args, time_array, folder)
        _write_collection_wedge(kmz_document, sicd, time_args, arp_llh, time_array, folder)


def add_sicd_from_ortho_helper(kmz_document, ortho_helper,
        inc_image_corners=False, inc_valid_data=False,
        inc_scp=False, inc_collection_wedge=False,
        block_size=10, remap_function=None):
    """
    Adds for a SICD to the provided open kmz from an ortho-rectification helper.

    Parameters
    ----------
    kmz_document : Document
    ortho_helper : OrthorectificationHelper
    inc_image_corners : bool
        Include the image corners, if possible?
    inc_valid_data : bool
        Include the valid image area, if possible?
    inc_scp : bool
        Include the scp?
    inc_collection_wedge : bool
        Include the aperture location and collection wedge?
    block_size : None|int|float
        The block size for the iterator
    remap_function : None|RemapFunction
        The remap function to apply, or a suitable default will be chosen.
    """

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(
            'ortho_helper must be an OrthorectificationHelper instance, got '
            'type {}'.format(type(ortho_helper)))
    if not isinstance(kmz_document, Document):
        raise TypeError(
            'kmz_document must be an sarpy.io.kml.Document instance, got '
            'type {}'.format(type(kmz_document)))

    # create a folder for these sicd details
    sicd = ortho_helper.sicd
    folder = kmz_document.add_container(
        the_type='Folder', name=_get_sicd_name(sicd), description=_get_sicd_description(sicd))

    # write the sicd details aside from the overlay
    add_sicd_geometry_elements(sicd, kmz_document, folder,
        inc_image_corners=inc_image_corners, inc_valid_data=inc_valid_data,
        inc_scp=inc_scp, inc_collection_wedge=inc_collection_wedge)

    # create the ortho-rectification iterator
    if remap_function is None:
        remap_function = NRL()
    calculator = FullResolutionFetcher(
        ortho_helper.reader, index=ortho_helper.index, dimension=1, block_size=block_size)
    ortho_iterator = OrthorectificationIterator(
        ortho_helper, calculator=calculator, remap_function=remap_function,
        recalc_remap_globals=True)

    # write the image overlay
    _write_sicd_overlay(ortho_iterator, kmz_document, folder)


def add_sicd_to_kmz(kmz_document, reader, index=0, pixel_limit=2048,
        inc_image_corners=False, inc_valid_data=False,
        inc_scp=False, inc_collection_wedge=False,
        block_size=10, remap_function=None):
    """
    Adds elements for this SICD to the provided open kmz.

    Parameters
    ----------
    kmz_document : Document
        The kmz document, which must be open and have an associated archive.
    reader : SICDTypeReader
        The reader instance, must be of sicd type:
    index : int
        The index to use.
    pixel_limit : None|int
        The limit in pixel size to use for the constructed ground overlay.
    inc_image_corners : bool
        Include the image corners, if possible?
    inc_valid_data : bool
        Include the valid image area, if possible?
    inc_scp : bool
        Include the scp?
    inc_collection_wedge : bool
        Include the aperture location and collection wedge?
    block_size : None|int|float
        The block size for the iterator
    remap_function : None|RemapFunction
        The remap function to apply, or a suitable default will be chosen.

    Returns
    -------
    None
    """

    if not isinstance(reader, SICDTypeReader):
        raise TypeError('reader must be a instance of SICDTypeReader. Got type {}'.format(type(reader)))

    if pixel_limit is not None:
        pixel_limit = int(pixel_limit)
        if pixel_limit < 512:
            pixel_limit = 512

    # create our projection helper
    index = int(index)
    sicd = reader.get_sicds_as_tuple()[index]
    try:
        proj_helper = PGRatPolyProjection(sicd)
    except SarpyRatPolyError:
        proj_helper = PGProjection(sicd)
    # create our orthorectification helper
    ortho_helper = NearestNeighborMethod(reader, index=index, proj_helper=proj_helper)
    if pixel_limit is not None:
        # let's see what the ortho-rectified size will be
        ortho_size = ortho_helper.get_full_ortho_bounds()
        row_count = ortho_size[1] - ortho_size[0]
        col_count = ortho_size[3] - ortho_size[2]
        # reset the row/column spacing, if necessary
        if row_count > pixel_limit:
            proj_helper.row_spacing *= row_count/float(pixel_limit)
        if col_count > pixel_limit:
            proj_helper.col_spacing *= col_count/float(pixel_limit)
        if isinstance(proj_helper, PGRatPolyProjection):
            proj_helper.perform_rational_poly_fitting()
    # add the sicd details
    add_sicd_from_ortho_helper(
        kmz_document, ortho_helper,
        inc_image_corners=inc_image_corners, inc_valid_data=inc_valid_data, inc_scp=inc_scp,
        inc_collection_wedge=inc_collection_wedge, block_size=block_size, remap_function=remap_function)


def create_kmz_view(
        reader, output_directory, file_stem='view', pixel_limit=2048,
        inc_image_corners=False, inc_valid_data=False,
        inc_scp=True, inc_collection_wedge=False, block_size=10, remap_function=None):
    """
    Create a kmz view for the reader contents. **This will create one file per
    band/polarization present in the reader.**

    Parameters
    ----------
    reader : SICDTypeReader
    output_directory : str
    file_stem : str
    pixel_limit : None|int
    inc_image_corners : bool
        Include the image corners, if possible?
    inc_valid_data : bool
        Include the valid image area, if possible?
    inc_scp : bool
        Include the scp?
    inc_collection_wedge : bool
        Include the aperture location and collection wedge?
    block_size : None|int|float
        The block size for the iterator
    remap_function : None|RemapFunction
        The remap function to apply, or a suitable default will be chosen.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        import logging
        logger = logging.getLogger('sarpy')
        logger.setLevel('INFO')

        import os
        from sarpy.io.complex.converter import open_complex
        from sarpy.io.product.kmz_product_creation import create_kmz_view

        test_root = '<root directory>'
        reader = open_complex(os.path.join(test_root, '<file name>>'))
        create_kmz_view(reader, test_root,
                        file_stem='View-<something descriptive>',
                        pixel_limit=2048,
                        inc_collection_wedge=True)

    """

    def get_pol_abbreviation(pol_in):
        spol = pol_in.split(':')
        if len(spol) == 2:
            return spol[0][0] + spol[1][0]
        return pol_in

    def do_iteration():
        kmz_file = os.path.join(output_directory, '{}_{}_{}.kmz'.format(file_stem,
                                                                        the_band,
                                                                        get_pol_abbreviation(the_pol)))
        logger.info('Writing kmz file for polarization {} and band {}'.format(the_pol, the_band))
        with prepare_kmz_file(kmz_file, name=reader.file_name) as kmz_doc:
            for the_partition, the_index, the_sicd in sicd_reader_iterator(
                    reader, partitions=partitions, polarization=the_pol, band=the_band):
                add_sicd_to_kmz(
                    kmz_doc, reader,
                    index=the_index, pixel_limit=pixel_limit,
                    inc_image_corners=inc_image_corners, inc_valid_data=inc_valid_data,
                    inc_scp=inc_scp, inc_collection_wedge=inc_collection_wedge,
                    block_size=block_size, remap_function=remap_function)

    bands = set(reader.get_sicd_bands())
    pols = set(reader.get_sicd_polarizations())
    partitions = reader.get_sicd_partitions()

    for the_band in bands:
        for the_pol in pols:
            do_iteration()
