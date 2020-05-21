# -*- coding: utf-8 -*-
"""
Functionality for exporting certain data elements to a kml document
"""

import zipfile
import logging
import sys
import os
from xml.dom import minidom
from typing import Union


if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    from cStringIO import StringIO
else:
    from io import StringIO

try:
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
        params : the parameters dictionary
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
        return self._doc.toprettyxml(encoding='utf-8')

    def _set_file(self, file_name):
        if isinstance(file_name, str):
            fext = os.path.splitext(file_name)
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

    def write_file_to_archive(self, arcpath, filepath):
        if self._archive is None:
            raise ValueError('No archive defined.')
        self._archive.write(filepath, arcpath, zipfile.ZIP_DEFLATED)

    def write_string_to_archive(self, arcpath, val):
        if self._archive is None:
            raise ValueError('No archive defined.')
        self._archive.writestr(zipfile.ZipInfo(arcpath), val, zipfile.ZIP_DEFLATED)

    def write_image_to_archive(self, arcpath, val, frmt='PNG'):
        imbuf = StringIO()
        val.save(imbuf, frmt)
        self.write_string_to_archive(arcpath, imbuf.getvalue())
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
        nod = self._doc.createElement(tag)
        nod.appendChild(self._doc.createTextNode(value))
        par.appendChild(nod)
        return nod

    def _add_cdata_node(self, par, tag, value):
        # type: (Union[None, minidom.Element], str, str) -> minidom.Element
        nod = self._doc.createElement(tag)
        nod.appendChild(self._doc.createCDATASection(value))
        par.appendChild(nod)
        return nod

    def _add_conditional_text_node(self, par, tag, params, default=None):
        # type: (Union[None, minidom.Element], str, dict, Union[None, str]) -> minidom.Element
        if tag in params:
            return self._add_text_node(par, tag, str(params[tag]))
        elif default is not None:
            return self._add_text_node(par, tag, str(default))

    def _add_conditional_cdata_node(self, par, tag, params, default=None):
        # type: (Union[None, minidom.Element], str, dict, Union[None, str]) -> minidom.Element
        if tag in params:
            return self._add_cdata_node(par, tag, str(params[tag]))
        elif default is not None:
            return self._add_cdata_node(par, tag, str(default))

    # basic kml container creation
    def add_container(self, par=None, typ='Placemark', **params):
        """
        For creation of Document, Folder, or Placemark container.

        Parameters
        ----------
        par : None|minidom.Element
        typ : str
            One of "Placemark", "Folder". The type "Document" can only be used once,
            for the top level construct.
        params : dict
            The dictionary of parameters

        Returns
        -------
        minidom.Element
        """

        if typ not in ("Placemark", "Folder", "Document"):
            raise ValueError('typ must be one of ("Placemark", "Folder", "Document")')

        container = self._create_new_node(par, typ)
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
            if 'beginTime' in params:
                self._add_text_node(ts, 'begin', params['beginTime'])
            if 'endTime' in params:
                self._add_text_node(ts, 'end', params['endTime'])
        elif 'when' in params:
            ts = self._create_new_node(container, 'TimeStamp')
            self._add_text_node(ts, 'when', params['when'])
        return container

    # Styles
    def add_style_map(self, sid, high_id, low_id):
        """
        Creates a styleMap from two given style ids.

        Parameters
        ----------
        sid : str
        high_id : str
        low_id : str

        Returns
        -------
        None
        """

        sm = self._create_new_node(None, 'StyleMap')
        sm.setAttribute('id', sid)

        pair1 = self._create_new_node(sm, 'Pair')
        self._add_text_node(pair1, 'key', 'normal')
        self._add_text_node(pair1, 'styleUrl', '#'+low_id)

        pair2 = self._create_new_node(sm, 'Pair')
        self._add_text_node(pair2, 'key', 'highlight')
        self._add_text_node(pair2, 'styleUrl', '#'+high_id)

    def add_style(self, pid, **params):
        """
        Creates a style for use in the document tree.

        Parameters
        ----------
        pid : str
            the style id string.
        params : dict
            the dictionary of the parameters

        Returns
        -------
        None
        """

        sty = self._create_new_node(None, 'Style')
        sty.setAttribute('id', pid)
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

    def add_line_style(self, pid=None, par=None, **params):
        """
        Add line style.

        Parameters
        ----------
        pid : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params : dict
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'LineStyle')
        if pid is not None:
            sty.setAttribute('id', pid)
        self._add_conditional_text_node(sty, 'color', params, default='b0ff0000')
        self._add_conditional_text_node(sty, 'width', params, default='1.0')

    def add_list_style(self, pid=None, par=None, **params):
        """
        Add list style

        Parameters
        ----------
        pid : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params : dict
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'ListStyle')
        if pid is not None:
            sty.setAttribute('id', pid)

        item_icon = self._create_new_node(sty, 'ItemIcon')
        if 'icnr' in params:
            self._add_text_node(item_icon, 'href', params['icnr'])
        else:
            self._add_text_node(item_icon, 'href', _DEFAULT_ICON)

    def add_label_style(self, pid=None, par=None, **params):
        """
        Add label style

        Parameters
        ----------
        pid : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params : dict
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'LabelStyle')
        if pid is not None:
            sty.setAttribute('id', pid)
        self._add_conditional_text_node(sty, 'color', params, default='b0ff0000')
        self._add_conditional_text_node(sty, 'scale', params, default='1.0')

    def add_icon_style(self, pid=None, par=None, **params):
        """
        Add icon style.

        Parameters
        ----------
        pid : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params : dict
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'IconStyle')
        if pid is not None:
            sty.setAttribute('id', pid)
        self._add_conditional_text_node(sty, 'color', params)
        self._add_conditional_text_node(sty, 'scale', params)
        icon = self._create_new_node(sty, 'Icon')
        if 'icnr' in params:
            self._add_text_node(icon, 'href', params['icnr'])
        else:
            self._add_text_node(icon, 'href', _DEFAULT_ICON)

    def add_poly_style(self, pid=None, par=None, **params):
        """
        Add poly style.

        Parameters
        ----------
        pid : None|str
            The id, which should not be set if this is a child of a style element.
        par : None|minidom.Element
            The parent node.
        params : dict
            The parameters dictionary.

        Returns
        -------
        None
        """

        sty = self._create_new_node(par, 'PolyStyle')
        if pid is not None:
            sty.setAttribute('id', pid)
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
            **{'line_style': line, 'label_style': label, 'icon_style': icon, 'poly_style': poly})

        line['width'] = '0.75'
        label['scale'] = '0.75'
        icon['scale'] = '0.75'
        self.add_style(
            'default_low',
            **{'line_style': line, 'label_style': label, 'icon_style': icon, 'poly_style': poly})
        self.add_style_map('defaultStyle', 'default_high', 'default_low')

    def add_color_ramp(self, cols, high_size=1.0, low_size=0.5, icnr=None, name_stem='sty'):
        """
        Adds collection of enumerated styles corresponding to provided array of colors.

        Parameters
        ----------
        cols : numpy.ndarray
            numpy array of shape (N, 4) of 8-bit colors assumed to be RGBA
        high_size : float
            The highlighted size.
        low_size : float
            The regular (low lighted?) size.
        icnr : str
            The icon reference.
        name_stem : str
            The string representing the naming convention for the associated styles.

        Returns
        -------
        None
        """

        hline = {'width': str(2.0*high_size)}
        hlabel = {'scale': str(high_size)}
        hicon = {'scale': str(high_size)}
        lline = {'width': str(2.0*low_size)}
        llabel = {'scale': str(low_size)}
        licon = {'scale': str(low_size)}
        if icnr is not None:
            hicon['icnr'] = icnr
            licon['icnr'] = icnr
        for i in range(cols.shape[0]):
            col = '{3:02x}{2:02x}{1:02x}{0:02x}'.format(*cols[i, :])
            for di in [hline, hlabel, hicon, lline, llabel, licon]:
                di['color'] = col
            self.add_style('{0!s}{1:d}_high'.format(name_stem, i),
                          **{'line_style': hline, 'label_style': hlabel, 'icon_style': hicon})
            self.add_style('{0!s}{1:d}_low'.format(name_stem, i),
                          **{'line_style': lline, 'label_style': llabel, 'icon_style': licon})
            self.add_style_map('{0!s}{1:d}'.format(name_stem, i),
                             '{0!s}{1:d}_high'.format(name_stem, i),
                             '{0!s}{1:d}_low'.format(name_stem, i))

    # extended data handling
    def add_schema(self, sid, fieldDict, sname=None):
        """
        For defining/using the extended data schema capability. **Note that this
        is specifically part of the google earth extension of kml, and may not be generally
        supported by anything except google earth.**

        Parameters
        ----------
        sid : str
            the schema id - must be unique to the id collection document
        fieldDict : dict
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
        sname : None|str
            optional short name for display in the schema

        Returns
        -------
        None
        """

        types = ['string', 'int', 'uint', 'short', 'ushort', 'float', 'double', 'bool']
        sch = self._create_new_node(None, 'Schema')
        sch.setAttribute('id', sid)
        if sname is not None:
            sch.setAttribute('name', sname)
        for name in fieldDict.keys():
            sf = self._doc.createElement('SimpleField')
            typ, dname = fieldDict[name]
            sf.setAttribute('name', name)
            if typ in types:
                sf.setAttribute('type', typ)
                sch.appendChild(sf)
            else:
                logging.warning("Schema '{0!s}' has field '{1!s}' of unrecognized "
                                "type '{2!s}', which is being excluded.".format(sid, name, typ))
            if dname is not None:
                self._add_text_node(sf, 'displayName', dname)

    def _add_extended_data(self, par, **params):
        """
        Adds ExtendedData (schema data) to the parent element. **Note that this
        is specifically part of the google earth extension of kml, and may not be generally
        supported by anything except google earth.**

        Parameters
        ----------
        par : minidom.Element
        params : dict
            the parameters dictionary

        Returns
        -------
        None
        """

        extended_data = self._create_new_node(par, 'ExtendedData')
        schema_data = self._create_new_node(extended_data, 'SchemaData')
        schema_data.setAttribute('schemaUrl', params['schemaUrl'])
        dat = params['ExtendedData']
        if 'fieldOrder' in params:
            keys = params['fieldOrder']
        else:
            keys = sorted(dat.keys())
        for key in keys:
            # check if data is iterable
            if hasattr(dat[key], '__iter__'):
                array_node = self._create_new_node(schema_data, 'gx:SimpleArrayData')
                array_node.setAttribute('name', key)
                for el in dat[key]:
                    self._add_text_node(array_node, 'gx:value', str(el))
            else:
                sid = self._add_text_node(schema_data, 'SimpleData', str(dat[key]))
                sid.setAttribute('name', key)

    # screen overlay
    def add_screen_overlay(self, im_name, par=None, **params):
        """
        Adds ScreenOverlay object.

        Parameters
        ----------
        im_name : str
            Reference to appropriate image object.
        par : None|minidom.Element
            The parent node. Appended at root level if not provided.
        params : dict
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
        self._add_text_node(ic, 'href', im_name)
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
        params : dict
            The parameters dictionary

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        multigeometry_node = self._create_new_node(par, 'MultiGeometry')
        return multigeometry_node

    def add_polygon(self, outCoords, inCoords=None, par=None, **params):
        """
        Adds a Polygon element - a polygonal outer region, possibly with polygonal holes removed

        Parameters
        ----------
        outCoords : str
            comma/space delimited string of coordinates for the outerRing. Format of the string
            :code:`'lon1,lat1,alt1 lon2,lat2,alt2 ...'` with the altitude values optional. If given, the altitude value
            is in meters. The precise interpretation of altitude (relative to the ground, relative to sea level, etc.)
            depends on the value of relevant tags passed down to the LinearRing objects, namely the values for the
            params entries:
                * 'extrude'
                * 'tessellate'
                * 'altitudeMode'
        inCoords : None|List[str]
            If provided, the coordinates for inner rings.
        par : None|minidom.Element
            The parent node. If not given, then a Placemark is created.
        params : dict
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(**params)
        polygon_node = self._create_new_node(par, 'Polygon')
        outer_ring_node = self._create_new_node(polygon_node, 'outerBoundaryIs')
        self.add_linear_ring(outCoords, outer_ring_node, **params)
        for coords in inCoords:
            inner_ring = self._create_new_node(polygon_node, 'innerBoundaryIs')
            self.add_linear_ring(coords, inner_ring, **params)

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
        params : dict
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(params=params)
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
        params : dict
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(params=params)
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
        params : dict
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(params=params)
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
        params : dict
            The parameters dictionary.

        Returns
        -------
            minidom.Element
        """

        if par is None:
            par = self.add_container(params=params)
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
        params : dict
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(params=params)
        gx_track = self._create_new_node(par, 'gx:Track')
        for opt in ['extrude', 'tessellate', 'altitudeMode']:
            self._add_conditional_text_node(gx_track, opt, params)
        for wh in whens:
            self._add_text_node(gx_track, 'when', wh)
        for coords in coords:
            self._add_text_node(gx_track, 'gx:coord', coords)
        if angles is not None:
            for an in angles:
                self._add_text_node(gx_track, 'gx:angles', '{0:0.8f} 0 0'.format(an))
        if ('ExtendedData' in params) and ('schemaUrl' in params):
            self._add_extended_data(gx_track, **params)
        return gx_track

    def add_ground_overlay(self, b_box, im_name, par=None, **params):
        """
        Adds GroundOverlay object.

        Parameters
        ----------
        b_box : List[str]
            list of the form: [latitude max, latitude min, longitude max, longitude min]
        im_name : str
            Reference to appropriate image object.
        par : None|minidom.Element
            The parent node. if not provided, then a Placemark object is created implicitly.
        params : dict
            The parameters dictionary.

        Returns
        -------
        minidom.Element
        """

        if par is None:
            par = self.add_container(params=params)
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
            if 'beginTime' in params:
                self._add_text_node(ts, 'begin', params['beginTime'])
            if 'endTime' in params:
                self._add_text_node(ts, 'end', params['endTime'])
        elif 'when' in params:
            ts = self._create_new_node(overlay, 'TimeStamp')
            self._add_conditional_text_node(ts, 'when', params)

        # latitude/longitude box parameters
        ll = self._create_new_node(overlay, 'LatLonBox')
        for cdir, num in zip(['north', 'south', 'east', 'west'], b_box):
            self._add_text_node(ll, cdir, str(num))
        self._add_conditional_text_node(ll, 'rotation', params)

        # icon
        ic = self._create_new_node(overlay, 'Icon')
        self._add_text_node(ic, 'href', im_name)
        return overlay

    # regionation - likely rarely used
    def _add_lod(self, par, **params):
        """
        Adds a Level of Detail (LOD) element, which is explicitly a child of Region.

        Parameters
        ----------
        par : minidom.Element
        params : dict

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
        params : dict

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
        params : dict

        Returns
        -------
        minidom.Element
        """

        reg = self._create_new_node(par, 'Region')
        self._add_lod(reg, **params)
        self._add_lat_lon_alt_box(reg, **params)
        return reg

    def add_regionated_ground_overlay(self, b_box, img, par, frmt='PNG', **params):
        """
        Adds regionated GroundOverlay objects. This downsamples the image to a pyramid type
        collection of smaller images, and defines the regions. **Requires viable archive.**

        Parameters
        ----------
        b_box : List[str]
            list of the form: :code:`[<latitude max>, <latitude min>, <longitude max>, <longitude min>]`,
            follows the [north, south, east, west] requirement for GroundOverlay objects
        img : PIL.Image.Image
            the image instance.
        par : minidom.Element
            the parent node, a folder object will be created and appended to par.
            The overlays will be added below this folder.
        frmt : str
            string representing a viable Image format. The viable options that will be allowed:
                * 'PNG' - (default) transparency; lossless; good compression
                * 'TIFF' - supports transparency; lossless; poor compression
                * 'JPEG' -  no transparency; lossy; best compression
                * 'GIF' - transparency; lossless; medium compression
            The PIL library actually supports a much larger collection of image formats, but the
            remaining formats are not good candidates for this application.
        params : dict
            The parameters dictionary.
        Returns
        -------
        minidom.Element
        """

        max_image_size = 2048
        min_image_size = 1024

        def split(ibox, cbox):
            """
            Recursive splitting.

            Parameters
            ----------
            ibox : List[int]
                left,upper,right,lower index bounding box
            cbox : List[float]
                north,south,east,west coordinate bounding box

            Returns
            -------
            None
            """

            wdth = (ibox[2] - ibox[0])
            hgt = (ibox[3] - ibox[1])
            if max(wdth, hgt) < (max_image_size * 3) / 2:
                # stopping criteria
                cont_recursion = False
                swdth = wdth
                shgt = hgt
            else:
                # resample and continue
                cont_recursion = True
                if wdth > hgt:
                    swdth = max_image_size
                    shgt = (hgt * max_image_size) / wdth
                else:
                    shgt = max_image_size
                    swdth = (wdth * max_image_size) / hgt
            # prepare to write image
            newimg = img.crop(ibox).resize((swdth, shgt), PIL.Image.ANTIALIAS)
            olName = params['name']
            global im_count
            fname = 'images/{}_{}.{}'.format(olName, im_count, frmt)
            pars = {'name': olName + '_' + str(im_count)}
            for key in ['beginTime', 'endTime']:
                if key in params:
                    pars[key] = params[key]
            p = self.add_ground_overlay(cbox, fname, par=fld, params=pars)
            pars = {}
            if im_count == 0:
                # root level, no minimum size
                pars['minLodPixels'] = 0
            else:
                pars['minLodPixels'] = min_image_size / 2
                pars['minFadeExtent'] = min_image_size / 2
            if cont_recursion:
                pars['maxLodPixels'] = max_image_size + min_image_size / 2
                pars['maxFadeExtent'] = min_image_size / 2
            else:
                # leaf, no maximum size
                pars['maxLodPixels'] = -1
            pars['north'] = cbox[0]
            pars['south'] = cbox[1]
            pars['east'] = cbox[2]
            pars['west'] = cbox[3]
            self.add_region(p, **pars)
            self.write_image_to_archive(fname, newimg, frmt=frmt)
            im_count += 1
            if cont_recursion:
                # determine correct subdivisions
                if wdth > (max_image_size * 3) / 2:
                    mid = wdth / 2
                    winds = [ibox[0], ibox[0] + wdth / 2, ibox[2]]
                    los = [cbox[3], cbox[3] + (cbox[2] - cbox[3]) * float(mid) / wdth, cbox[2]]
                else:
                    winds = [ibox[0], ibox[2]]
                    los = [cbox[3], cbox[2]]

                if hgt > (max_image_size * 3) / 2:
                    mid = hgt / 2
                    hinds = [ibox[1], ibox[1] + hgt / 2, ibox[3]]
                    las = [cbox[0], cbox[0] - (cbox[0] - cbox[1]) * float(mid) / hgt, cbox[1]]
                else:
                    hinds = [ibox[1], ibox[3]]
                    las = [cbox[0], cbox[1]]
                # call recursion
                for j in range(len(winds) - 1):
                    for k in range(len(hinds) - 1):
                        split((winds[j], hinds[k], winds[j + 1], hinds[k + 1]),
                              (las[k], las[k + 1], los[j + 1], los[j]))
            return

        if self._archive is None:
            raise ValueError('We must have a viable archive.')
        if PIL is None:
            raise ImportError('We cannot import the PIL library, so this functionality cannot be used.')
        if not isinstance(img, PIL.Image.Image):
            raise TypeError('We must have that img is a PIL instance.')

        # create our folder object
        fld = self.add_container(par, typ='Folder', params=params)
        # check the image size
        verticalPixels = img.size[1]
        horizontalPixels = img.size[0]

        if frmt not in ['PNG', 'TIFF', 'JPEG', 'GIF']:
            logging.warning("Invalid image format {} replaced with PNG".format(frmt))
            frmt = 'PNG'

        global im_count
        im_count = 0

        # call the recursion
        split((0, 0, horizontalPixels, verticalPixels), b_box)
