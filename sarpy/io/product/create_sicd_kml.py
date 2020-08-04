# -*- coding: utf-8 -*-
"""
This module provides tools for creating kmz visualizations of a SICD.
"""

from typing import Union
from xml.dom import minidom
import json

import numpy

from sarpy.compliance import int_func, integer_types
from sarpy.io.general.base import BaseReader
from sarpy.processing.ortho_rectify import OrthorectificationHelper, \
    NearestNeighborMethod, PGProjection, OrthorectificationIterator
from sarpy.io.kml import Document
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.geometry.geocoords import ecf_to_geodetic

try:
    # noinspection PyPackageRequirements
    import PIL
    import PIL.Image
except ImportError:
    PIL = None


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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

    # bounding box style - basic polygon, probably clamped to ground
    line = {'color': 'ccff5050', 'width': '2.0'}
    poly = {'color': '30ff5050'}
    kmz_document.add_style('bounding_high', line_style=line, poly_style=poly)
    line['width'] = '1.0'
    kmz_document.add_style('bounding_low', line_style=line, poly_style=poly)
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
    icon = {'color': 'ff5050c0', 'scale': '1.0',
            'icon_ref': 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'}
    kmz_document.add_style('scp_high', label_style=label, icon_style=icon)
    label['scale'] = '0.75'
    icon['scale'] = '0.5'
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

    return 'ortho-rectified image<br>' \
           'row resolution - {0:0.2f} meters<br>' \
           'column resolution - {1:0.2f} meters'.format(
        ortho_iterator.ortho_helper.proj_helper.row_spacing,
        ortho_iterator.ortho_helper.proj_helper.col_spacing)


def _get_sicd_time_args(sicd, subdivisions=12):
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

    end_time = beg_time + int_func(sicd.Timeline.CollectDuration*1e6)
    if not isinstance(subdivisions, integer_types) or subdivisions < 2:
        time_array = None
    else:
        time_array = numpy.linspace(0, sicd.Timeline.CollectDuration, subdivisions)
    return {'beginTime': str(beg_time)+'Z', 'endTime': str(end_time)+'Z'}, time_array


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
    coords = ['{1:0.8f},{0:0.8f},{2:0.2f}'.format(*el) for el in arp_llh]
    whens = [str(sicd.Timeline.CollectStart.astype('datetime64[us]') + int_func(el*1e6)) + 'Z' for el in time_array]
    placemark = kmz_document.add_container(par=folder, description='aperture pos', styleUrl='#arp', **time_args)
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
    coord_array = [frm.format(*el) for el in arp_llh]
    if len(grp_llh) > 1:
        coord_array.extend(frm.format(*el) for el in grp_llh[::-1, :])
    else:
        coord_array.append(frm.format(*grp_llh[0, :]))
    coord_array.append(frm.format(*arp_llh[0, :]))
    coords = ' '.join(coord_array)
    placemark = kmz_document.add_container(par=folder, description='collection wedge', styleUrl='#collection', **time_args)
    kmz_document.add_polygon(coords, par=placemark, extrude=False, tesselate=False, altitudeMode='absolute')


def _write_sicd_geometry_elements(sicd, kmz_document, folder):
    """
    Write the geometry elements of a SICD.

    Parameters
    ----------
    sicd : SICDType
    kmz_document : Document
    folder : minidom.Element

    Returns
    -------
    None
    """

    # let's define the time data for the SICD
    time_args, time_array = _get_sicd_time_args(sicd)

    if sicd.GeoData is not None:
        # add the image corners/bounding box
        frm = '{1:0.8f},{0:0.8f},0'
        if sicd.GeoData.ImageCorners is not None:
            corners = sicd.GeoData.ImageCorners.get_array(dtype='float64')
            coords = ' '.join(frm.format(*el) for el in corners)
            coords += ' ' + frm.format(*corners[0, :])
            placemark = kmz_document.add_container(par=folder, description='image corners', styleUrl='#bounding')
            kmz_document.add_polygon(coords, par=placemark, altitudeMode='clampToGround', **time_args)
        # add the valid data
        if sicd.GeoData.ValidData is not None:
            valid_array = sicd.GeoData.ValidData.get_array(dtype='float64')
            coords = ' '.join(frm.format(*el) for el in valid_array)
            coords += ' ' + frm.format(*valid_array[0, :])
            placemark = kmz_document.add_container(par=folder, description='valid data', styleUrl='#valid')
            kmz_document.add_polygon(coords, par=placemark, altitudeMode='clampToGround', **time_args)
        # write scp data
        if sicd.GeoData.SCP is not None:
            coords = frm.format(*sicd.GeoData.SCP.LLH.get_array())
            placemark = kmz_document.add_container(par=folder, description='SCP', styleUrl='#scp')
            kmz_document.add_point(coords, par=placemark, altitudeMode='clampToGround', **time_args)
    # write arp position
    arp_llh = _write_arp_location(kmz_document, sicd, time_args, time_array, folder)
    # write collection wedge
    _write_collection_wedge(kmz_document, sicd, time_args, arp_llh, time_array, folder)


def _write_sicd_overlay(ortho_iterator, kmz_document, folder):
    def reorder_corners(llh_in):
        return llh_in[::-1, :]

    time_args, _ = _get_sicd_time_args(ortho_iterator.sicd, subdivisions=None)

    # create the output workspace
    image_data = numpy.zeros(ortho_iterator.ortho_data_size, dtype='uint8')
    # populate by iterating
    for data, start_indices in ortho_iterator:
        image_data[
        start_indices[0]:start_indices[0]+data.shape[0],
        start_indices[1]:start_indices[1]+data.shape[1]] = data
    # create regionated overlay
    # convert image array to PIL image.
    img = PIL.Image.fromarray(image_data.T)
    lat_lon_quad = reorder_corners(ortho_iterator.get_llh_image_corners())
    kmz_document.add_regionated_ground_overlay(
        img, folder, lat_lon_quad=lat_lon_quad[:, :2], img_format='JPEG',
        name='image overlay', description=_get_orthoiterator_description(ortho_iterator))


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


def add_sicd_from_ortho_helper(kmz_document, ortho_helper):
    """
    Adds for a SICD to the provided open kmz from an ortho-rectification helper.

    Parameters
    ----------
    kmz_document : Document
    ortho_helper : OrthorectificationHelper

    Returns
    -------
    None
    """

    if PIL is None:
        raise ImportError(
            'This functionality cannot be used with the optional Pillow dependency.')

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
    _write_sicd_geometry_elements(sicd, kmz_document, folder)
    # write the image overlay
    ortho_iterator = OrthorectificationIterator(ortho_helper)
    _write_sicd_overlay(ortho_iterator, kmz_document, folder)


def add_sicd_to_kmz(kmz_document, reader, index=0):
    """
    Adds elements for this SICD to the provided open kmz.

    Parameters
    ----------
    kmz_document : Document
        The kmz document, which must be open and have an associated archive.
    reader : BaseReader
        The reader instance, must be of sicd type:
    index : int
        The index to use.

    Returns
    -------
    None
    """

    if not isinstance(reader, BaseReader):
        raise TypeError('reader must be a instance of BaseReader. Got type {}'.format(type(reader)))
    if not reader.is_sicd_type:
        raise ValueError('reader must be of sicd type.')

    # create our projection helper
    index = int(index)
    sicd = reader.get_sicds_as_tuple()[index]
    proj_helper = PGProjection(sicd)  # , row_spacing=5, col_spacing=5)
    # TODO: we should set appropriate row and column spacing for the projection helper
    #   to have some moderately sized sicd (at most 2048 or 4096 pixels on the given side?)
    # create our orthorectification helper
    ortho_helper = NearestNeighborMethod(reader, index=index, proj_helper=proj_helper)
    # add the sicd details
    add_sicd_from_ortho_helper(kmz_document, ortho_helper)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level='INFO')
    import os
    from sarpy.io.complex.converter import open_complex
    test_root = os.path.expanduser('~/Desktop/sarpy_testing/sicd')
    # open our sicd file
    reader = open_complex(os.path.join(test_root, 'sicd_example_RMA_RGZERO_RE16I_IM16I.nitf'))
    # ortho_helper = NearestNeighborMethod(reader, index=0)

    # prepare our kmz document
    out_file = os.path.join(test_root, 'test1.kmz')
    kmz_doc = prepare_kmz_file(out_file, name='test')
    with kmz_doc:
        add_sicd_to_kmz(kmz_doc, reader)
