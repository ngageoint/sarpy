# -*- coding: utf-8 -*-
"""
Functionality for exporting certain data elements to a kml document
"""

import zipfile
import logging
import os
import numpy
from xml.dom import minidom
from typing import Union, List
from uuid import uuid4

from sarpy.compliance import BytesIO, string_types, int_func
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic

try:
    # noinspection PyPackageRequirements
    import PIL
    # noinspection PyPackageRequirements
    import PIL.Image
except ImportError:
    PIL = None

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

#################
# default values
_DEFAULT_ICON = 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'


class Document(object):
    """
    The main kml document container, and zip archive if the output file is
    of type kmz. *This is intended to be used as a context manager.*
    """

    __slots__ = ('_doc', '_document', '_archive', '_file', '_closed')

    def __init__(self, file_name=None, **params):
        """

        Parameters
        ----------
        file_name : str|zipfile.ZipFile|file like
            The output location or buffer to which to write the kml/other objects
        params
            The parameters dictionary for file creation.
        """

        self._file = None
        self._archive = None
        self._closed = False
        self._set_file(file_name)

        self._doc = minidom.Document()
        kml = self._doc.createElement('kml')
        self._doc.appendChild(kml)
        kml.setAttribute('xmlns', 'http://www.opengis.net/kml/2.2')
        kml.setAttribute('xmlns:gx', 'http://www.google.com/kml/ext/2.2')
        kml.setAttribute('xmlns:kml', 'http://www.opengis.net/kml/2.2')
        kml.setAttribute('xmlns:atom', 'http://www.w3.org/2005/Atom')
        self._document = self.add_container(kml, 'Document', **params)

    def __str__(self):
        xml = self._doc.toprettyxml(encoding='utf-8')
        if not isinstance(xml, string_types):
            return xml.decode('utf-8')
        else:
            return xml

    def _set_file(self, file_name):
        if isinstance(file_name, str):
            fext = os.path.splitext(file_name)[1]
            if fext not in ['.kml', '.kmz']:
                logging.warning('file extension should be one of .kml or .kmz, got {}. This will be treated as a kml file.'.format(fext))
            if fext == '.kmz':
                self._archive = zipfile.ZipFile(file_name, 'w', zipfile.ZIP_DEFLATED)
            else:
                self._file = open(file_name, 'w')
        elif isinstance(file_name, zipfile.ZipFile):
            self._archive = file_name
        elif hasattr(file_name, 'write'):
            self._file = file_name
        else:
            raise TypeError('file_name must be a file path, file-like object, or a zipfile.Zipfile instance')

    def close(self):
        if self._closed:
            return

        if self._file is not None:
            self._file.write(str(self))
            self._file.close()
        else:
            self.write_string_to_archive('doc.kml', str(self))
            self._archive.close()
        self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None:
            self.close()
        else:
            logging.error(
                'The kml/kmz file writer generated an exception during processing. Any generated file '
                'may be only partially generated and/or corrupt.')
            # The exception will be reraised - it's unclear how any exception could be caught.

    def write_file_to_archive(self, archive_path, file_path):
        """
        Copy the given file into the kmz archive at the given archive location.

        Parameters
        ----------
        archive_path : str
            The location in the archive.
        file_path : str
            The file location on the file system.

        Returns
        -------
        None
        """

        if self._archive is None:
            raise ValueError('No archive defined.')
        self._archive.write(file_path, archive_path, zipfile.ZIP_DEFLATED)

    def write_string_to_archive(self, archive_path, val):
        """
        Write the given string/bytes into the kmz archive at the given location.

        Parameters
        ----------
        archive_path : str
        val : bytes|str

        Returns
        -------
        None
        """

        if self._archive is None:
            raise ValueError('No archive defined.')
        self._archive.writestr(zipfile.ZipInfo(archive_path), val, zipfile.ZIP_DEFLATED)

    def write_image_to_archive(self, archive_path, val, img_format='PNG'):
        """
        Write the given PIL image into the kmz archive at the given location.

        Parameters
        ----------
        archive_path : str
        val : PIL.Image.Image
        img_format : str

        Returns
        -------
        None
        """

        imbuf = BytesIO()
        val.save(imbuf, img_format)
        self.write_string_to_archive(archive_path, imbuf.getvalue())
        imbuf.close()

    # xml node creation elements
    def _create_new_node(self, par, tag):
        # type: (Union[None, minidom.Element], str) -> minidom.Element
        nod = self._doc.createElement(tag)
        if par is None:
            self._document.appendChild(nod)
        else:
            par.appendChild(nod)
        return nod

    def _add_text_node(self, par, tag, value):
        # type: (Union[None, minidom.Element], str, str) -> minidom.Element
        if value is None:
            return

        nod = self._doc.createElement(tag)
        if isinstance(value, string_types):
            nod.appendChild(self._doc.createTextNode(value))
        else:
            nod.appendChild(self._doc.createTextNode(str(value)))
        par.appendChild(nod)
        return nod

    def _add_cdata_node(self, par, tag, value):
        # type: (Union[None, minidom.Element], str, str) -> minidom.Element
        if value is None:
            return

        nod = self._doc.createElement(tag)
        if isinstance(value, string_types):
            nod.appendChild(self._doc.createCDATASection(value))
        else:
            nod.appendChild(self._doc.createCDATASection(str(value)))
        par.appendChild(nod)
        return nod

    def _add_conditional_text_node(self, par, tag, params, default=None):
        # type: (Union[None, minidom.Element], str, dict, Union[None, str]) -> minidom.Element

        return self._add_text_node(par, tag, params.get(tag, default))

    def _add_conditional_cdata_node(self, par, tag, params, default=None):
        # type: (Union[None, minidom.Element], str, dict, Union[None, str]) -> minidom.Element
        return self._add_cdata_node(par, tag, params.get(tag, default))

    # basic kml container creation
    def add_container(self, par=None, the_type='Placemark', **params):
        """
        For creation of Document, Folder, or Placemark container.

        Parameters
        ----------
        par : None|minidom.Element
        the_type : str
            One of "Placemark", "Folder". The type "Document" can only be used once,
            for the top level construct.
        params
            The dictionary of parameters

        Returns
        -------
        minidom.Element
        """

        if the_type not in ("Placemark", "Folder", "Document"):
            raise ValueError('the_type must be one of ("Placemark", "Folder", "Document")')

        container = self._create_new_node(par, the_type)
        if 'id' in params:
            container.setAttribute('id', params['id'])

        for opt in ['name', 'Snippet', 'styleUrl']:
            self._add_conditional_text_node(container, opt, params)
        self._add_conditional_cdata_node(container, 'description', params)

        # extended data
        if ('schemaUrl' in params) and ('ExtendedData' in params):
            self._add_extended_data(container, **params)
        if ('beginTime' in params) or ('endTime' in params):
            ts = self._create_new_node(container, 'TimeSpan')
            self._add_text_node(ts, 'begin', params.get('beginTime', None))
            self._add_text_node(ts, 'end', params.get('endTime', None))
        elif 'when' in params:
            ts = self._create_new_node(container, 'TimeStamp')
            self._add_text_node(ts, 'when', params['when'])
        return container

    # Styles
    def add_style_map(self, style_id, high_id, low_id):
        """
        Creates a styleMap from two given style ids.

        Parameters
        ----------
        style_id : str
        high_id : str
        low_id : str

        Returns
        -------
        None
        """

        sm = self._create_new_node(None, 'StyleMap')
        sm.setAttribute('id', style_id)

        pair1 = self._create_new_node(sm, 'Pair')
        self._add_text_node(pair1, 'key', 'normal')
        self._add_text_node(pair1, 'styleUrl', '#'+low_id)

        pair2 = self._create_new_node(sm, 'Pair')
        self._add_text_node(pair2, 'key', 'highlight')
        self._add_text_node(pair2, 'styleUrl', '#'+high_id)

    def add_style(self, style_id, **params):
        """
        Creates a style for use in the document tree.

        Parameters
        ----------
        style_id : str
            the style id string.
        params
            the dictionary of the parameters

        Returns
        -------
        None
        """

        sty = self._create_new_node(None, 'Style')
        sty.setAttribute('id', style_id)
        if 'line_style' in params:
            self.add_line_style(None, sty, **params['line_style'])
        if 'label_style' in params:
            self.add_label_style(None, sty, **params['label_style'])
        if 'list_style' in params:
            self.add_list_style(None, sty, **params['list_style'])
        if 'icon_style' in params:
            self.add_icon_style(None, sty, **params['icon_style'])
        if 'poly_style' in params:
            self.add_poly_style(None, sty, **params['poly_style'])

    def add_line_style(self, style_id=None, par=None, **params):
        """
        Add line style.

        Parameters
        ----------
        style_id : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'LineStyle')
        if style_id is not None:
            sty.setAttribute('id', style_id)
        self._add_conditional_text_node(sty, 'color', params, default='b0ff0000')
        self._add_conditional_text_node(sty, 'width', params, default='1.0')

    def add_list_style(self, style_id=None, par=None, **params):
        """
        Add list style

        Parameters
        ----------
        style_id : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'ListStyle')
        if style_id is not None:
            sty.setAttribute('id', style_id)

        item_icon = self._create_new_node(sty, 'ItemIcon')
        self._add_text_node(item_icon, 'href', params.get('icon_ref', _DEFAULT_ICON))

    def add_label_style(self, style_id=None, par=None, **params):
        """
        Add label style

        Parameters
        ----------
        style_id : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'LabelStyle')
        if style_id is not None:
            sty.setAttribute('id', style_id)
        self._add_conditional_text_node(sty, 'color', params, default='b0ff0000')
        self._add_conditional_text_node(sty, 'scale', params, default='1.0')

    def add_icon_style(self, style_id=None, par=None, **params):
        """
        Add icon style.

        Parameters
        ----------
        style_id : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'IconStyle')
        if style_id is not None:
            sty.setAttribute('id', style_id)
        self._add_conditional_text_node(sty, 'color', params)
        self._add_conditional_text_node(sty, 'scale', params)
        icon = self._create_new_node(sty, 'Icon')
        self._add_text_node(icon, 'href', params.get('icon_ref', _DEFAULT_ICON))

    def add_poly_style(self, style_id=None, par=None, **params):
        """
        Add poly style.

        Parameters
        ----------
        style_id : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'PolyStyle')
        if style_id is not None:
            sty.setAttribute('id', style_id)
        self._add_conditional_text_node(sty, 'color', params, default='80ff0000')
        self._add_conditional_text_node(sty, 'fill', params)
        self._add_conditional_text_node(sty, 'outline', params)

    def add_default_style(self):
        """
        Add default style

        The style is created, and appended at root level. The corresponding styleUrl is '#defaultStyle'
        """

        line = {'color': 'ff505050', 'width': '1.0'}
        label = {'color': 'ffc0c0c0', 'scale': '1.0'}
        icon = {'color': 'ffff5050', 'scale': '1.0'}
        poly = {'color': '80ff5050'}
        self.add_style(
            'default_high',
            line_style=line, label_style=label, icon_style=icon, poly_style=poly)

        line['width'] = '0.75'
        label['scale'] = '0.75'
        icon['scale'] = '0.75'
        self.add_style(
            'default_low',
            line_style=line, label_style=label, icon_style=icon, poly_style=poly)
        self.add_style_map('defaultStyle', 'default_high', 'default_low')

    def add_color_ramp(self, colors, high_size=1.0, low_size=0.5, icon_ref=None, name_stem='sty'):
        """
        Adds collection of enumerated styles corresponding to provided array of colors.

        Parameters
        ----------
        colors : numpy.ndarray
            numpy array of shape (N, 4) of 8-bit colors assumed to be RGBA
        high_size : float
            The highlighted size.
        low_size : float
            The regular (low lighted?) size.
        icon_ref : str
            The icon reference.
        name_stem : str
            The string representing the naming convention for the associated styles.

        Returns
        -------
        None
        """

        hline = {'width': 2*high_size}
        hlabel = {'scale': high_size}
        hicon = {'scale': high_size}
        lline = {'width': 2*low_size}
        llabel = {'scale': low_size}
        licon = {'scale': low_size}
        if icon_ref is not None:
            hicon['icon_ref'] = icon_ref
            licon['icon_ref'] = icon_ref
        for i in range(colors.shape[0]):
            col = '{3:02x}{2:02x}{1:02x}{0:02x}'.format(*colors[i, :])
            for di in [hline, hlabel, hicon, lline, llabel, licon]:
                di['color'] = col
            self.add_style(
                '{0!s}{1:d}_high'.format(name_stem, i),
                line_style=hline, label_style=hlabel, icon_style=hicon)
            self.add_style(
                '{0!s}{1:d}_low'.format(name_stem, i),
                line_style=lline, label_style=llabel, icon_style=licon)
            self.add_style_map(
                '{0!s}{1:d}'.format(name_stem, i),
                '{0!s}{1:d}_high'.format(name_stem, i),
                '{0!s}{1:d}_low'.format(name_stem, i))

    # extended data handling
    def add_schema(self, schema_id, field_dict, short_name=None):
        """
        For defining/using the extended data schema capability. **Note that this
        is specifically part of the google earth extension of kml, and may not be generally
        supported by anything except google earth.**

        Parameters
        ----------
        schema_id : str
            the schema id - must be unique to the id collection document
        field_dict : dict
            dictionary where the key is field name. The corresponding value is a tuple of the form
            `(type, displayName)`, where `displayName` can be `None`. The value of `type` is one of
            the data types permitted:
                * 'string'
                * 'int
                * 'uint'
                * 'short'
                * 'ushort'
                * 'float'
                * 'double'
                * 'bool'
        short_name : None|str
            optional short name for display in the schema

        Returns
        -------
        None
        """

        types = ['string', 'int', 'uint', 'short', 'ushort', 'float', 'double', 'bool']
        sch = self._create_new_node(None, 'Schema')
        sch.setAttribute('id', schema_id)
        if short_name is not None:
            sch.setAttribute('name', short_name)
        for name in field_dict:
            sf = self._doc.createElement('SimpleField')
            typ, dname = field_dict[name]
            sf.setAttribute('name', name)
            if typ in types:
                sf.setAttribute('type', typ)
                sch.appendChild(sf)
            else:
                logging.warning("Schema '{0!s}' has field '{1!s}' of unrecognized "
                                "type '{2!s}', which is being excluded.".format(schema_id, name, typ))
            self._add_text_node(sf, 'displayName', dname)

    def _add_extended_data(self, par, **params):
        """
        Adds ExtendedData (schema data) to the parent element. **Note that this
        is specifically part of the google earth extension of kml, and may not be generally
        supported by anything except google earth.**

        Parameters
        ----------
        par : minidom.Element
        params
            the parameters dictionary

        Returns
        -------
        None
        """

        extended_data = self._create_new_node(par, 'ExtendedData')
        schema_data = self._create_new_node(extended_data, 'SchemaData')
        schema_data.setAttribute('schemaUrl', params['schemaUrl'])
        dat = params['ExtendedData']
        keys = params.get('fieldOrder', sorted(dat.keys()))

        for key in keys:
            # check if data is iterable
            if hasattr(dat[key], '__iter__'):
                array_node = self._create_new_node(schema_data, 'gx:SimpleArrayData')
                array_node.setAttribute('name', key)
                for el in dat[key]:
                    self._add_text_node(array_node, 'gx:value', el)
            else:
                sid = self._add_text_node(schema_data, 'SimpleData', dat[key])
                sid.setAttribute('name', key)

    def add_screen_overlay(self, image_ref, par=None, **params):
        """
        Adds ScreenOverlay object.

        Parameters
        ----------
        image_ref : str
            Reference to appropriate image object, whether in the kmz archive or
            an appropriate url.
        par : None|minidom.Element
            The parent node. Appended at root level if not provided.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        overlay = self._create_new_node(par, 'ScreenOverlay')
        if 'id' in params:
            overlay.setAttribute('id', params['id'])

        for opt in ['name', 'Snippet', 'styleUrl', 'rotation']:
            self._add_conditional_text_node(overlay, opt, params)
        self._add_conditional_cdata_node(overlay, 'description', params)

        # extended data
        if ('schemaUrl' in params) and ('ExtendedData' in params):
            self._add_extended_data(overlay, **params)

        # overlay parameters
        for opt in ['overlayXY', 'screenXY', 'size', 'rotationXY']:
            olp = self._doc.createElement(opt)
            good = True
            for att in ['x', 'y', 'xunits', 'yunits']:
                key = '{}:{}'.format(opt, att)
                if key in params:
                    olp.setAttribute(att, params[key])
                else:
                    logging.error(
                        'params is missing required key {}, so we are aborting screen '
                        'overlay parameters construction. This screen overlay will likely '
                        'not render correctly.'.format(key))
                    good = False
            if good:
                overlay.appendChild(olp)
        # icon
        ic = self._create_new_node(overlay, 'Icon')
        self._add_text_node(ic, 'href', image_ref)
        return overlay

    # direct kml geometries
    def add_multi_geometry(self, par=None, **params):
        """
        Adds a MultiGeometry object. The MultiGeometry object is really just a container.
        The user must continue adding the primitive Geometry constituents to this container or
        nothing will actually get rendered.

        Parameters
        ----------
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        multigeometry_node = self._create_new_node(par, 'MultiGeometry')
        return multigeometry_node

    def add_polygon(self, outer_coords, inner_coords=None, par=None, **params):
        """
        Adds a Polygon element - a polygonal outer region, possibly with polygonal holes removed

        Parameters
        ----------
        outer_coords : str
            comma/space delimited string of coordinates for the outerRing. Format of the string
            :code:`'lon1,lat1,alt1 lon2,lat2,alt2 ...'` with the altitude values optional. If given, the altitude value
            is in meters. The precise interpretation of altitude (relative to the ground, relative to sea level, etc.)
            depends on the value of relevant tags passed down to the LinearRing objects, namely the values for the
            params entries:
                * 'extrude'
                * 'tessellate'
                * 'altitudeMode'
        inner_coords : None|List[str]
            If provided, the coordinates for inner rings.
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        polygon_node = self._create_new_node(par, 'Polygon')
        for opt in ['extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(polygon_node, opt, params)

        outer_ring_node = self._create_new_node(polygon_node, 'outerBoundaryIs')
        self.add_linear_ring(outer_coords, outer_ring_node)
        if inner_coords is not None:
            for coords in inner_coords:
                inner_ring = self._create_new_node(polygon_node, 'innerBoundaryIs')
                self.add_linear_ring(coords, inner_ring)

    def add_linear_ring(self, coords, par=None, **params):
        """
        Adds a LinearRing element (closed linear path).

        Parameters
        ----------
        coords : str
            comma/space delimited string of coordinates for the outerRing. Format of the string
            :code:`'lon1,lat1,alt1 lon2,lat2,alt2 ...'` with the altitude values optional. If given, the altitude value
            is in meters. The precise interpretation of altitude (relative to the ground, relative to sea level, etc.)
            depends on the value of relevant tags passed down to the LinearRing objects, namely the values for the
            params entries:
                * 'extrude'
                * 'tessellate'
                * 'altitudeMode'
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        linear_ring = self._create_new_node(par, 'LinearRing')

        for opt in ['extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(linear_ring, opt, params)
        self._add_text_node(linear_ring, 'coordinates', coords)
        return linear_ring

    def add_line_string(self, coords, par=None, **params):
        """
        Adds a LineString element (linear path).

        Parameters
        ----------
        coords : str
            comma/space delimited string of coordinates for the outerRing. Format of the string
            :code:`'lon1,lat1,alt1 lon2,lat2,alt2 ...'` with the altitude values optional. If given, the altitude value
            is in meters. The precise interpretation of altitude (relative to the ground, relative to sea level, etc.)
            depends on the value of relevant tags passed down to the LinearRing objects, namely the values for the
            params entries:
                * 'extrude'
                * 'tessellate'
                * 'altitudeMode'
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        line_string = self._create_new_node(par, 'LineString')

        for opt in ['extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(line_string, opt, params)
        self._add_text_node(line_string, 'coordinates', coords)
        return line_string

    def add_point(self, coords, par=None, **params):
        """
        Adds a Point object.

        Parameters
        ----------
        coords : str
            comma/space delimited string of coordinates for the outerRing. Format of the string
            :code:`'lon1,lat1,alt1 lon2,lat2,alt2 ...'` with the altitude values optional. If given, the altitude value
            is in meters. The precise interpretation of altitude (relative to the ground, relative to sea level, etc.)
            depends on the value of relevant tags passed down to the LinearRing objects, namely the values for the
            params entries:
                * 'extrude'
                * 'tessellate'
                * 'altitudeMode'
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        point = self._create_new_node(par, 'Point')

        for opt in ['extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(point, opt, params)
        self._add_text_node(point, 'coordinates', coords)

    def add_gx_multitrack(self, par=None, **params):
        """
        Adds a MultiTrack from the gx namespace. This is only a container, much like
        a MultiGeometry object, which requires the addition of gx:Track objects. **Note that this
        is specifically part of the google earth extension of kml, and may not be generally
        supported by anything except google earth.**

        Parameters
        ----------
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary.

        Returns
        -------
            minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        gx_multitrack = self._create_new_node(par, 'gx:MultiTrack')

        for opt in ['gx:interpolate', 'extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(gx_multitrack, opt, params)
        return gx_multitrack

    def add_gx_track(self, coords, whens, angles=None, par=None, **params):
        """
        Adds a Track from the gx namespace. **Note that this
        is specifically part of the google earth extension of kml, and may not be generally
        supported by anything except google earth.**

        Parameters
        ----------
        coords : List[str]
            list of comma delimited string of coordinates. Format of each string entry: 'lon1,lat1,alt1'
            with altitude values optional. If given, the altitude value is in meters. The precise
            interpretation of altitude (relative to the ground, relative to sea level, etc.) depends on
            the value of relevant tags passed down to the LinearRing objects, namely the values for the
            params entries:
                * 'extrude'
                * 'tessellate'
                * 'altitudeMode'
        whens : List[str]
            list of iso-formatted time strings - entries matching coords
        angles : None|List[str]
            None or list of heading (rotation) angles for the icon. If None, then Google Earth
            infers from the path.
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        gx_track = self._create_new_node(par, 'gx:Track')
        for opt in ['extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(gx_track, opt, params)
        for wh in whens:
            self._add_text_node(gx_track, 'when', wh)
        for coords in coords:
            self._add_text_node(gx_track, 'gx:coord', coords)
        if angles is not None:
            for an in angles:
                self._add_text_node(gx_track, 'gx:angles', '{} 0 0'.format(an))
        if ('ExtendedData' in params) and ('schemaUrl' in params):
            self._add_extended_data(gx_track, **params)
        return gx_track

    def add_ground_overlay(self, image_ref, bounding_box=None, lat_lon_quad=None, par=None, **params):
        """
        Adds GroundOverlay object, defined either from a bounding_box or a lat/lon
        quadrilateral.

        Parameters
        ----------
        image_ref : str
            Reference to appropriate image object, either in the kmz archive or
            an appropriate url.
        bounding_box : None|numpy.ndarray|tuple|list
            list of the form `[latitude max, latitude min, longitude max, longitude min]`
        lat_lon_quad : None|numpy.ndarray|list|tuple
            list of the form [[latitude, longitude]], must have 4 entries. The orientation
            is counter-clockwise from the lower-left image corner.
        par : None|minidom.Element
            The parent node. if not provided, then a Placemark object is created implicitly.
        params
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if bounding_box is None and lat_lon_quad is None:
            raise ValueError('Either bounding_box or lat_lon_quad must be defined.')
        if bounding_box is not None and lat_lon_quad is not None:
            raise ValueError('Both bounding_box or lat_lon_quad are provided, which is not sensible.')

        if par is None:
            par = self.add_container(**params)
        overlay = self._create_new_node(par, 'GroundOverlay')
        if 'id' in params:
            overlay.setAttribute('id', params['id'])
        for opt in ['name', 'Snippet', 'styleUrl', 'altitude', 'altitudeMode']:
            self._add_conditional_text_node(overlay, opt, params)
        self._add_conditional_cdata_node(overlay, 'description', params)

        # extended data
        if ('schemaUrl' in params) and ('ExtendedData' in params):
            self._add_extended_data(overlay, **params)

        # time parameters
        if ('beginTime' in params) or ('endTime' in params):
            ts = self._create_new_node(overlay, 'TimeSpan')
            self._add_text_node(ts, 'begin', params.get('beginTime', None))
            self._add_text_node(ts, 'end', params.get('endTime', None))
        elif 'when' in params:
            ts = self._create_new_node(overlay, 'TimeStamp')
            self._add_conditional_text_node(ts, 'when', params)

        if bounding_box is not None:
            # latitude/longitude box parameters
            ll = self._create_new_node(overlay, 'LatLonBox')
            for cdir, num in zip(['north', 'south', 'east', 'west'], bounding_box):
                self._add_text_node(ll, cdir, num)
            self._add_conditional_text_node(ll, 'rotation', params)
        elif lat_lon_quad is not None:
            if len(lat_lon_quad) != 4:
                raise ValueError('lat_lon_quad must have length 4.')
            # latitude/longitude quad parameters
            llq = self._create_new_node(overlay, 'gx:LatLonQuad')
            coords = ''
            for entry in lat_lon_quad:
                if isinstance(entry, string_types):
                    coords += entry.strip() + ' '
                elif len(entry) >= 2:
                    coords += '{0:0.8f},{1:0.8f} '.format(entry[1], entry[0])
                else:
                    raise TypeError('Got unexpected entry type {}'.format(type(entry)))
            self._add_text_node(llq, 'coordinates', coords.strip())

        # icon
        ic = self._create_new_node(overlay, 'Icon')
        self._add_text_node(ic, 'href', image_ref)
        return overlay

    # regionation tools for ground overlays
    def _add_lod(self, par, **params):
        """
        Adds a Level of Detail (LOD) element, which is explicitly a child of Region.

        Parameters
        ----------
        par : minidom.Element
        params

        Returns
        -------
        minidom.Element
        """

        lod = self._create_new_node(par, 'Lod')
        self._add_conditional_text_node(lod, 'minLodPixels', params, '128')
        self._add_conditional_text_node(lod, 'maxLodPixels', params, '-1')
        self._add_conditional_text_node(lod, 'minFadeExtent', params, '0')
        return lod

    def _add_lat_lon_alt_box(self, par, **params):
        """
        Adds LatLonAltBox element, which is explicitly a child of Region.

        Parameters
        ----------
        par : minidom.Element
        params

        Returns
        -------
        minidom.Element
        """

        box = self._create_new_node(par, 'LatLonAltBox')
        for key in ['north', 'south', 'east', 'west', 'minAltitude', 'maxAltitude', 'altitudeMode']:
            self._add_conditional_text_node(box, key, params)
        return box

    def add_region(self, par, **params):
        """
        Adds a region element.

        Parameters
        ----------
        par : None|minidom.Element
        params

        Returns
        -------
        minidom.Element
        """

        reg = self._create_new_node(par, 'Region')
        self._add_lod(reg, **params)
        self._add_lat_lon_alt_box(reg, **params)
        return reg

    def _add_ground_overlay_region_bbox(
            self, image_name, fld, img, image_bounds, bounding_box,
            nominal_image_size, img_format, depth_count=0, **params):
        """
        Helper function for creating an ground overlay region part.

        Parameters
        ----------
        image_name : str
            The image name.
        fld : minidom.Element
        img : PIL.Image.Image
        image_bounds : numpy.ndarray|list|tuple
            Using PIL conventions, of the form `(col min, row min, col max, row max)`.
        bounding_box : numpy.ndarray|tuple|list
            Bounding box of the form `[latitude max, latitude min, longitude max, longitude min]`
        nominal_image_size : int
        img_format : str
        depth_count : int
            What is the depth for our recursion.
        params
            The parameters dictionary.

        Returns
        -------
        None
        """

        col_min, row_min, col_max, row_max = image_bounds

        # determine how to resample this image
        row_length = int_func(row_max - row_min)
        col_length = int_func(col_max - col_min)
        cont_recursion = True
        if max(row_length, col_length) < 1.5*nominal_image_size:
            cont_recursion = False
            sample_rows = row_length
            sample_cols = col_length
        elif row_length >= col_length:
            sample_rows = nominal_image_size
            sample_cols = int_func(col_length*nominal_image_size/float(row_length))
        else:
            sample_cols = nominal_image_size
            sample_rows = int_func(row_length*nominal_image_size/float(col_length))

        archive_name = 'images/{}.{}'.format(image_name, img_format)
        # resample our image
        pil_box = tuple(int_func(el) for el in image_bounds)
        this_img = img.crop(pil_box).resize((sample_cols, sample_rows), PIL.Image.ANTIALIAS)
        self.write_image_to_archive(archive_name, this_img, img_format=img_format)
        # create the ground overlay parameters
        pars = {'name': image_name}
        for key in ['beginTime', 'endTime', 'when']:
            if key in params:
                pars[key] = params[key]
        # create the ground overlay
        gnd_overlay = self.add_ground_overlay(archive_name, bounding_box=bounding_box, par=fld, **pars)
        # add the region
        pars = {}
        if depth_count == 0:
            # root level, no minimum size
            pars['minLodPixels'] = 0
        else:
            pars['minLodPixels'] = 0.3*nominal_image_size
            pars['minFadeExtent'] = 0.3*nominal_image_size
        if cont_recursion:
            pars['maxLodPixels'] = 1.75*nominal_image_size
            pars['maxFadeExtent'] = 0.3*nominal_image_size
        else:
            # leaf, no maximum size
            pars['maxLodPixels'] = -1
        pars['north'] = bounding_box[0]
        pars['south'] = bounding_box[1]
        pars['east'] = bounding_box[2]
        pars['west'] = bounding_box[3]
        self.add_region(gnd_overlay, **pars)

        if cont_recursion:
            # create a list of [(start row, end row)]
            if row_length > 1.5*nominal_image_size:
                split_row = row_min + int_func(0.5*row_length)
                split_lat = bounding_box[0] + (split_row/float(row_length))*(bounding_box[1] - bounding_box[0])
                row_sizes = [(row_min, split_row), (split_row, row_max)]
                lats = [(bounding_box[0], split_lat), (split_lat, bounding_box[1])]
            else:
                row_sizes = [(row_min, row_max), ]
                lats = [(bounding_box[0], bounding_box[1]), ]

            if col_length > 1.5*nominal_image_size:
                split_col = col_min + int_func(0.5*col_length)
                split_lon = bounding_box[2] + (split_col/float(row_length))*(bounding_box[3] - bounding_box[2])
                col_sizes = [(col_min, split_col), (split_col, col_max)]
                lons = [(bounding_box[2], split_lon), (split_lon, bounding_box[3])]
            else:
                col_sizes = [(col_min, col_max), ]
                lons = [(bounding_box[2], bounding_box[3]), ]

            count = 0
            for row_bit, lat_bit in zip(row_sizes, lats):
                for col_bit, lon_bit in zip(col_sizes, lons):
                    this_im_name = '{}_{}'.format(image_name, count)
                    this_im_bounds = row_bit + col_bit
                    this_bounding_box = lat_bit + lon_bit
                    self._add_ground_overlay_region_bbox(
                        this_im_name, fld, img, this_im_bounds, this_bounding_box,
                        nominal_image_size, img_format, depth_count=depth_count+1, **params)
                    count += 1

    @staticmethod
    def _split_lat_lon_quad(ll_quad, split_fractions):
        """
        Helper method for recursively splitting the lat/lon quad box.

        Parameters
        ----------
        ll_quad : numpy.ndarray
        split_fractions : list|tuple

        Returns
        -------
        numpy.ndarray
        """

        r1, r2, c1, c2 = split_fractions
        # [0] corresponds to (max_row, 0)
        # [1] corresponds to (max row, max_col)
        # [2] corresponds to (0, max_col)
        # [3] corresponds to (0, 0)

        # do row split
        # [0] = r2*[0] + (1-r2)*[3]
        # [1] = r2*[1] + (1-r2)*[2]
        # [2] = r1*[1] + (1-r1)*[2]
        # [3] = r1*[0] + (1-r1)*[3]
        row_split = numpy.array([
            [r2, 0, 0, 1-r2],
            [0, r2, 1-r2, 0],
            [0, r1, 1-r1, 0],
            [r1, 0, 0, 1-r1],
        ], dtype='float64')

        # do column split
        # [0] = (1-c1)*[0] + c1*[1]
        # [1] = (1-c2)*[0] + c2*[1]
        # [2] = c2*[2] + (1-c2)*[3]
        # [3] = c1*[2] + (1-c1)*[3]
        col_split = numpy.array([
            [1-c1, c1, 0, 0],
            [1-c2, c2, 0, 0],
            [0, 0, c2, 1-c2],
            [0, 0, c1, 1-c1],], dtype='float64')

        split = col_split.dot(row_split)

        llh_temp = numpy.zeros((4, 3))
        llh_temp[:, :2] = ll_quad
        ecf_coords = geodetic_to_ecf(llh_temp)
        split_ecf = split.dot(ecf_coords)
        return ecf_to_geodetic(split_ecf)[:, :2]

    def _add_ground_overlay_region_quad(
            self, image_name, fld, img, image_bounds, lat_lon_quad,
            nominal_image_size, img_format, depth_count=0, **params):
        """
        Helper function for creating an ground overlay region part.

        Parameters
        ----------
        image_name : str
            The image name.
        fld : minidom.Element
        img : PIL.Image.Image
        image_bounds : numpy.ndarray|list|tuple
            Using PIL conventions, of the form `(col min, row min, col max, row max)`.
        lat_lon_quad : numpy.ndarray
            list of the form [[latitude, longitude]], must have 4 entries.
        nominal_image_size : int
        img_format : str
        depth_count : int
            What is the depth of the recursion?
        params

        Returns
        -------
        None
        """

        bounding_box = [
            float(numpy.max(lat_lon_quad[:, 0])), float(numpy.min(lat_lon_quad[:, 0])),
            float(numpy.max(lat_lon_quad[:, 1])), float(numpy.min(lat_lon_quad[:, 1]))]

        col_min, row_min, col_max, row_max = image_bounds

        # determine how to resample this image
        row_length = int_func(row_max - row_min)
        col_length = int_func(col_max - col_min)
        cont_recursion = True
        if max(row_length, col_length) <= 1.5*nominal_image_size:
            cont_recursion = False
            sample_rows = row_length
            sample_cols = col_length
        elif row_length >= col_length:
            sample_rows = nominal_image_size
            sample_cols = int_func(col_length*nominal_image_size/float(row_length))
        else:
            sample_cols = nominal_image_size
            sample_rows = int_func(row_length*nominal_image_size/float(col_length))

        logging.info('Processing ({}:{}, {}:{}) into a downsampled image of size ({}, {})'.format(
            row_min, row_max, col_min, col_max, sample_rows, sample_cols))

        archive_name = 'images/{}.{}'.format(image_name, img_format)
        pil_box = tuple(int_func(el) for el in image_bounds)
        # resample our image
        this_img = img.crop(pil_box).resize((sample_cols, sample_rows), PIL.Image.ANTIALIAS)
        self.write_image_to_archive(archive_name, this_img, img_format=img_format)
        # create the ground overlay parameters
        pars = {'name': image_name}
        for key in ['beginTime', 'endTime', 'when']:
            if key in params:
                pars[key] = params[key]
        # create the ground overlay
        gnd_overlay = self.add_ground_overlay(archive_name, lat_lon_quad=lat_lon_quad, par=fld, **pars)
        # add the region
        pars = {}
        if depth_count == 0:
            # root level, no minimum size
            pars['minLodPixels'] = 0
        else:
            pars['minLodPixels'] = 0.3*nominal_image_size
            pars['minFadeExtent'] = 0.3*nominal_image_size

        if cont_recursion:
            pars['maxLodPixels'] = 1.75*nominal_image_size
            pars['maxFadeExtent'] = 0.3*nominal_image_size
        else:
            # leaf, no maximum size
            pars['maxLodPixels'] = -1

        pars['north'] = bounding_box[0]
        pars['south'] = bounding_box[1]
        pars['east'] = bounding_box[2]
        pars['west'] = bounding_box[3]
        self.add_region(gnd_overlay, **pars)

        if cont_recursion:
            if row_length >= 1.5*nominal_image_size:
                split_row = row_min + int_func(0.5*row_length)
                row_sizes = [(row_min, split_row), (split_row, row_max)]
            else:
                row_sizes = [(row_min, row_max), ]

            if col_length >= 1.5*nominal_image_size:
                split_col = col_min + int_func(0.5*col_length)
                col_sizes = [(col_min, split_col), (split_col, col_max)]
            else:
                col_sizes = [(col_min, col_max), ]

            count = 0
            for row_bit in enumerate(row_sizes):
                for col_bit in enumerate(col_sizes):
                    this_im_name = '{}_{}'.format(image_name, count)
                    this_im_bounds = (col_bit[1][0], row_bit[1][0], col_bit[1][1], row_bit[1][1])
                    split_fractions = [
                        (row_bit[1][0] - row_min)/float(row_length),
                        (row_bit[1][1] - row_min)/float(row_length),
                        (col_bit[1][0] - col_min)/float(col_length),
                        (col_bit[1][1] - col_min)/float(col_length)
                    ]

                    this_ll_quad = self._split_lat_lon_quad(lat_lon_quad, split_fractions)

                    self._add_ground_overlay_region_quad(
                        this_im_name, fld, img, this_im_bounds, this_ll_quad,
                        nominal_image_size, img_format, depth_count=depth_count+1,
                        **params)
                    count += 1

    def add_regionated_ground_overlay(
            self, img, par, bounding_box=None, lat_lon_quad=None, img_format='PNG',
            nominal_image_size=1024, **params):
        """
        Adds regionated GroundOverlay objects. This downsamples the image to a pyramid type
        collection of smaller images, and defines the regions. **Requires viable archive.**

        Parameters
        ----------
        img : PIL.Image.Image
            the image instance.
        par : minidom.Element
            the parent node, a folder object will be created and appended to par.
            The overlays will be added below this folder.
        bounding_box : None|numpy.ndarray
            Follows the format for the argument in :func:`add_ground_overlay`.
        lat_lon_quad : None|nunpy.ndarray
            Follows the format for the argument in :func:`add_ground_overlay`.
        img_format : str
            string representing a viable Image format. The viable options that will be allowed:
                * 'PNG' - (default) transparency; lossless; good compression
                * 'TIFF' - supports transparency; lossless; poor compression
                * 'JPEG' -  no transparency; lossy; best compression
                * 'GIF' - transparency; lossless; medium compression
            The PIL library actually supports a much larger collection of image formats, but the
            remaining formats are not good candidates for this application.
        nominal_image_size : int
            The nominal image size for splitting. A minimum of 512 will be enforced.
        params
            The parameters dictionary.
        Returns
        -------
        minidom.Element
        """

        nominal_image_size = int_func(nominal_image_size)
        if nominal_image_size < 512:
            nominal_image_size = 512

        if self._archive is None:
            raise ValueError('We must have a viable archive.')
        if PIL is None:
            raise ImportError(
                'Optional dependency Pillow is required to use this functionality.')
        if not isinstance(img, PIL.Image.Image):
            raise TypeError('We must have that img is a PIL instance, got type {}.'.format(type(img)))

        # validate ground overlay area arguments
        if bounding_box is None and lat_lon_quad is None:
            raise ValueError('Either bounding_box or lat_lon_quad must be defined.')
        if bounding_box is not None and lat_lon_quad is not None:
            raise ValueError('Both bounding_box or lat_lon_quad are provided, which is not sensible.')
        if lat_lon_quad is not None:
            if not isinstance(lat_lon_quad, numpy.ndarray) or lat_lon_quad.ndim != 2 or \
                    lat_lon_quad.shape[0] != 4 or lat_lon_quad.shape[1] != 2:
                raise TypeError('lat_lon_quad, if supplied, must be a numpy array of shape (4, 2).')

        # create our folder object
        fld = self.add_container(par, the_type='Folder', **params)
        # get base name
        base_img_name = '{}-image'.format(uuid4())
        base_img_box = (0, 0, img.size[0], img.size[1])

        if bounding_box is not None:
            self._add_ground_overlay_region_bbox(
                base_img_name, fld, img, base_img_box, bounding_box,
                nominal_image_size, img_format, **params)
        elif lat_lon_quad is not None:
            self._add_ground_overlay_region_quad(
                base_img_name, fld, img, base_img_box, lat_lon_quad,
                nominal_image_size, img_format, **params)
