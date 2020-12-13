# -*- coding: utf-8 -*-
"""
The data extension header element definition.
"""

import logging
from typing import Union

from .base import BaseNITFElement, NITFElement, Unstructured, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor, \
    _parse_str, _parse_int, _parse_nitf_element
from .security import NITFSecurityTags, NITFSecurityTags0

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class XMLDESSubheader(NITFElement):
    """
    The standard XML Data Extension user header used in SICD and SIDD, described
    in SICD standard 2014-09-30, Volume II, page 29
    """

    _ordering = (
        'DESSHL', 'DESCRC', 'DESSHFT', 'DESSHDT',
        'DESSHRP', 'DESSHSI', 'DESSHSV', 'DESSHSD',
        'DESSHTN', 'DESSHLPG', 'DESSHLPT', 'DESSHLI',
        'DESSHLIN', 'DESSHABS')
    _lengths = {
        'DESSHL': 4, 'DESCRC': 5, 'DESSHFT': 8, 'DESSHDT': 20,
        'DESSHRP': 40, 'DESSHSI': 60, 'DESSHSV': 10,
        'DESSHSD': 20, 'DESSHTN': 120, 'DESSHLPG': 125,
        'DESSHLPT': 25, 'DESSHLI': 20, 'DESSHLIN': 120,
        'DESSHABS': 200}
    DESSHFT = _StringDescriptor(
        'DESSHFT', True, 8, default_value='XML',
        docstring='XML File Type. Data in this field shall be representative of the XML File Type. '
                  'Examples :code:`XSD, XML, DTD, XSL, XSLT`.')  # type: str
    DESSHDT = _StringDescriptor(
        'DESSHDT', True, 20, default_value='',
        docstring='Date and Time. This field shall contain the time (UTC) of the XML files '
                  'origination in the format :code:`YYYY-MM-DDThh:mm:ssZ`.')  # type: str
    DESSHRP = _StringDescriptor(
        'DESSHRP', True, 40, default_value='',
        docstring='Responsible Party – Organization Identifier. Identification of the '
                  'organization responsible for the content of the DES.')  # type: str
    DESSHSI = _StringDescriptor(
        'DESSHSI', True, 60, default_value='',
        docstring='Specification Identifier. Name of the specification used for the '
                  'XML data content.')  # type: str
    DESSHSV = _StringDescriptor(
        'DESSHSV', True, 10, default_value='',
        docstring='Specification Version. Version or edition of the specification.')  # type: str
    DESSHSD = _StringDescriptor(
        'DESSHSD', True, 20, default_value='',
        docstring='Specification Date. Version or edition date for the specification '
                  'in the format :code:`YYYY-MM-DDThh:mm:ssZ`.')  # type: str
    DESSHTN = _StringDescriptor(
        'DESSHTN', True, 120, default_value='',
        docstring='Target Namespace. Identification of the target namespace, if any, '
                  'designated within the XML data content.')  # type: str
    DESSHLPG = _StringDescriptor(
        'DESSHLPG', True, 125, default_value='',
        docstring='Location – Polygon. Five-point boundary enclosing the area applicable to the '
                  'DES, expressed as the closed set of coordinates of the polygon (last point '
                  'replicates first point). **NOTE** This is only an approximate reference so '
                  'specifying the coordinate reference system is unnecessary.\n'
                  'Recorded as paired latitude and longitude values in decimal degrees with '
                  'no separator. Each latitude and longitude value includes an explicit :code:`+` '
                  'or :code:`-`.\n'
                  'The precision for recording the values in the subheader is dictated by the field '
                  'size constraint.')  # type: str
    DESSHLPT = _StringDescriptor(
        'DESSHLPT', True, 25, default_value='',
        docstring='Location – Point. Single geographic point applicable to the DES.')  # type: str
    DESSHLI = _StringDescriptor(
        'DESSHLI', True, 20, default_value='',
        docstring='Location – Identifier. Identifier used to represent a geographic area. An '
                  'alphanumeric value identifying an instance in the designated namespace. When '
                  'this field is recorded with other than the default value, the Location Identifier '
                  'Namespace URI shall also be recorded.')  # type: str
    DESSHLIN = _StringDescriptor(
        'DESSHLIN', True, 120, default_value='',
        docstring='Location Identifier Namespace URI. URI for the Namespace where the Location '
                  'Identifier is described.')  # type: str
    DESSHABS = _StringDescriptor(
        'DESSHABS', True, 200, default_value='',
        docstring='Abstract. Brief narrative summary of the content of the DES.')  # type: str

    def __init__(self, **kwargs):
        self._DESSHL = 773
        self._DESCRC = 99999
        super(XMLDESSubheader, self).__init__(**kwargs)

    @property
    def DESSHL(self):
        """
        int: User defined subheader length
        """

        return self._DESSHL

    @DESSHL.setter
    def DESSHL(self, value):
        self._DESSHL = 773

    @property
    def DESCRC(self):
        """
        int: Cyclic redundancy check code, or 99999 when CRC not calculated/used.
        """

        return self._DESCRC

    @DESCRC.setter
    def DESCRC(self, value):
        self._DESCRC = 99999


##########
# DES - NITF 2.1 version

class DESUserHeader(Unstructured):
    _size_len = 4


class DataExtensionHeader(NITFElement):
    """
    The data extension subheader - see standards document MIL-STD-2500C for more
    information.
    """

    _ordering = ('DE', 'DESID', 'DESVER', 'Security', 'DESOFLW', 'DESITEM', 'UserHeader')
    _lengths = {'DE': 2, 'DESID': 25, 'DESVER': 2, 'DESOFLW': 6, 'DESITEM': 3}
    DE = _StringEnumDescriptor(
        'DE', True, 2, {'DE', }, default_value='DE',
        docstring='File part type.')  # type: str
    DESVER = _IntegerDescriptor(
        'DESVER', True, 2, default_value=1,
        docstring='Version of the Data Definition. This field shall contain the alphanumeric '
                  'version number of the use of the tag. The version number is assigned as '
                  'part of the registration process.')  # type: int
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags, default_args={},
        docstring='The security tags.')  # type: NITFSecurityTags

    def __init__(self, **kwargs):
        self._DESID = None
        self._DESOFLW = None
        self._DESITEM = None
        self._UserHeader = None
        super(DataExtensionHeader, self).__init__(**kwargs)

    @property
    def DESID(self):
        """
        str: Unique DES Type Identifier. This field shall contain a valid alphanumeric
        identifier properly registered with the ISMC.
        """

        return self._DESID

    @DESID.setter
    def DESID(self, value):
        value = _parse_str(value, 25, 'XML_DATA_CONTENT', 'DESID', self)
        self._DESID = value
        if value == 'TRE_OVERFLOW':
            if self.DESOFLW is None:
                self._DESOFLW = ''
            if self.DESITEM is None:
                self._DESITEM = 0
        else:
            self._DESOFLW = None
            self._DESITEM = None

    @property
    def DESOFLW(self):
        """
        None|str: DES Overflowed Header Type. This field shall be populated if
        `DESID = "TRE_OVERFLOW"`.

        Its presence indicates that the DES contains a TRE that would not fit in the file
        header or segment subheader where it would ordinarily be located. Its value indicates
        the segment type to which the enclosed TRE is relevant. If populated, must be one of
        :code:`{"XHD", "IXSHD", "SXSHD", "TXSHD", "UDHD", "UDID"}`.
        """

        return self._DESOFLW

    @DESOFLW.setter
    def DESOFLW(self, value):
        value = _parse_str(value, 6, None, 'DESOFLW', self)
        if self._DESID == 'TRE_OVERFLOW':
            if value is None:
                logging.error(
                    'DESOFLW value is None, but DESID == "TRE_OVERFLOW". '
                    'This must be resolved.')
                self._DESOFLW = ''
            elif value not in {'XHD', 'IXSHD', 'SXSHD', 'TXSHD', 'UDHD', 'UDID'}:
                logging.error(
                    "DESOFLW value got {}, but must be one {'XHD', 'IXSHD', 'SXSHD', 'TXSHD', 'UDHD', 'UDID'}"
                    "This must be resolved.")
                self._DESOFLW = ''
            else:
                self._DESOFLW = value
        else:
            if value is not None:
                logging.error(
                    'DESID != "TRE_OVERFLOW", but DESOFLW value is not None. '
                    'This is invalid, so setting DESOFLW to None')
            self._DESOFLW = None

    @property
    def DESITEM(self):
        """
        None|int: DES Data Item Overflowed. This field shall be present if `DESOFLW` is present.
        It shall contain the number of the data item in the file, of the type indicated in
        `DESOFLW` to which the TRE in the segment apply.
        """

        return self._DESITEM

    @DESITEM.setter
    def DESITEM(self, value):
        value = _parse_int(value, 3, None, 'DESITEM', self)
        if self._DESID == 'TRE_OVERFLOW':
            if value is None:
                logging.error(
                    'DESITEM value is None, but DESID == "TRE_OVERFLOW". '
                    'This must be resolved.')
                self._DESITEM = 0
            else:
                self._DESITEM = value
        else:
            if value is not None:
                logging.error(
                    'DESID != "TRE_OVERFLOW", but DESITEM value is not None. '
                    'This is invalid, so setting DESITEM to None')
            self._DESITEM = None

    @property
    def UserHeader(self):  # type: () -> Union[DESUserHeader, XMLDESSubheader]
        """
        DESUserHeader: The DES user header.
        """

        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        if not isinstance(value, BaseNITFElement):
            value = _parse_nitf_element(value, DESUserHeader, {}, 'UserHeader', self)
        self._UserHeader = value
        self._load_header_data()

    def _load_header_data(self):
        """
        Load any user defined header specifics.

        Returns
        -------
        None
        """

        if not isinstance(self._UserHeader, DESUserHeader):
            return

        if self.DESID.strip() == 'XML_DATA_CONTENT':
            # try loading sicd
            if self._UserHeader.get_bytes_length() == 777:
                # It could be a version 1.0 or greater SICD
                data = self._UserHeader.to_bytes()
                try:
                    data = XMLDESSubheader.from_bytes(data, 0)
                    self._UserHeader = data
                except Exception as e:
                    logging.error(
                        'DESID is "XML_DATA_CONTENT" and data is the right length for SICD, '
                        'but parsing failed with error {}'.format(e))
        elif self.DESID.strip() == 'STREAMING_FILE_HEADER':
            # LOW Priority - I think that this is deprecated?
            pass

    def _get_attribute_length(self, fld):
        if fld == 'DESOFLW':
            return 0 if self._DESOFLW is None else self._lengths['DESOFLOW']
        elif fld == 'DESITEM':
            return 0 if self._DESITEM is None else self._lengths['DESITEM']
        else:
            return super(DataExtensionHeader, self)._get_attribute_length(fld)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'UserHeader':
            val = DESUserHeader.from_bytes(value, start)
            fields['UserHeader'] = val
            return start + val.get_bytes_length()
        elif attribute == 'DESID':
            val = value[start:start+cls._lengths['DESID']].decode('utf-8')
            fields['DESID'] = val
            if val.strip() != 'TRE_OVERFLOW':
                fields['DESOFLW'] = None
                fields['DESITEM'] = None
            return start+cls._lengths['DESID']
        else:
            return super(DataExtensionHeader, cls)._parse_attribute(fields, attribute, value, start)


##########
# DES - NITF 2.0 version

class DataExtensionHeader0(NITFElement):
    """
    The data extension subheader - see standards document MIL-STD-2500C for more
    information.
    """

    _ordering = ('DE', 'DESTAG', 'DESVER', 'Security', 'DESOFLW', 'DESITEM', 'UserHeader')
    _lengths = {'DE': 2, 'DESTAG': 25, 'DESVER': 2, 'DESOFLW': 6, 'DESITEM': 3}
    DE = _StringEnumDescriptor(
        'DE', True, 2, {'DE', }, default_value='DE',
        docstring='File part type.')  # type: str
    DESVER = _IntegerDescriptor(
        'DESVER', True, 2, default_value=1,
        docstring='Version of the Data Definition. This field shall contain the alphanumeric '
                  'version number of the use of the tag. The version number is assigned as '
                  'part of the registration process.')  # type: int
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags0, default_args={},
        docstring='The security tags.')  # type: NITFSecurityTags0

    def __init__(self, **kwargs):
        self._DESTAG = None
        self._DESOFLW = None
        self._DESITEM = None
        self._UserHeader = None
        super(DataExtensionHeader0, self).__init__(**kwargs)

    @property
    def DESTAG(self):
        """
        str: Unique DES Type Identifier. This field shall contain a valid alphanumeric
        identifier properly registered with the ISMC.
        """

        return self._DESTAG

    @DESTAG.setter
    def DESTAG(self, value):
        value = _parse_str(value, 25, 'XML_DATA_CONTENT', 'DESTAG', self)
        self._DESTAG = value
        if value.strip() in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']:
            if self.DESOFLW is None:
                self._DESOFLW = ''
            if self.DESITEM is None:
                self._DESITEM = 0
        else:
            self._DESOFLW = None
            self._DESITEM = None

    @property
    def DESOFLW(self):
        """
        None|str: DES Overflowed Header Type. This field shall be populated if
        `DESTAG in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']`.

        Its presence indicates that the DES contains a TRE that would not fit in the file
        header or segment subheader where it would ordinarily be located. Its value indicates
        the segment type to which the enclosed TRE is relevant. If populated, must be one of
        :code:`{"XHD", "IXSHD", "SXSHD", "TXSHD", "UDHD", "UDID"}`.
        """

        return self._DESOFLW

    @DESOFLW.setter
    def DESOFLW(self, value):
        value = _parse_str(value, 6, None, 'DESOFLW', self)
        if self._DESTAG.strip() in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']:
            if value is None:
                logging.error(
                    'DESOFLW value is None, but DESTAG in [TRE_OVERFLOW, Registered Extensions, Controlled Extensions]. '
                    'This must be resolved.')
                self._DESOFLW = ''
            elif value not in {'XHD', 'IXSHD', 'SXSHD', 'TXSHD', 'UDHD', 'UDID'}:
                logging.error(
                    "DESOFLW value got {}, but must be one {'XHD', 'IXSHD', 'SXSHD', 'TXSHD', 'UDHD', 'UDID'}"
                    "This must be resolved.")
                self._DESOFLW = ''
            else:
                self._DESOFLW = value
        else:
            if value is not None:
                logging.error(
                    'DESTAG not in [TRE_OVERFLOW, Registered Extensions, Controlled Extensions], '
                    'but DESOFLW value is not None. This is invalid, so setting DESOFLW to None')
            self._DESOFLW = None

    @property
    def DESITEM(self):
        """
        None|int: DES Data Item Overflowed. This field shall be present if `DESOFLW` is present.
        It shall contain the number of the data item in the file, of the type indicated in
        `DESOFLW` to which the TRE in the segment apply.
        """

        return self._DESITEM

    @DESITEM.setter
    def DESITEM(self, value):
        value = _parse_int(value, 3, None, 'DESITEM', self)
        if self._DESTAG.strip() in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']:
            if value is None:
                logging.error(
                    'DESITEM value is None, but DESTAG in [TRE_OVERFLOW, Registered Extensions, Controlled Extensions]. '
                    'This must be resolved.')
                self._DESITEM = 0
            else:
                self._DESITEM = value
        else:
            if value is not None:
                logging.error(
                    'DESTAG not in [TRE_OVERFLOW, Registered Extensions, Controlled Extensions], '
                    'but DESITEM value is not None. This is invalid, so setting DESITEM to None')
            self._DESITEM = None

    @property
    def UserHeader(self):  # type: () -> Union[DESUserHeader, XMLDESSubheader]
        """
        DESUserHeader: The DES user header.
        """

        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        if not isinstance(value, BaseNITFElement):
            value = _parse_nitf_element(value, DESUserHeader, {}, 'UserHeader', self)
        self._UserHeader = value
        self._load_header_data()

    def _load_header_data(self):
        """
        Load any user defined header specifics.

        Returns
        -------
        None
        """

        if not isinstance(self._UserHeader, DESUserHeader):
            return

        if self.DESTAG.strip() == 'XML_DATA_CONTENT':
            # try loading sicd
            if self._UserHeader.get_bytes_length() == 777:
                # It could be a version 1.0 or greater SICD
                data = self._UserHeader.to_bytes()
                try:
                    data = XMLDESSubheader.from_bytes(data, 0)
                    self._UserHeader = data
                except Exception as e:
                    logging.error(
                        'DESTAG is "XML_DATA_CONTENT" and data is the right length for SICD, '
                        'but parsing failed with error {}'.format(e))
        elif self.DESTAG.strip() == 'STREAMING_FILE_HEADER':
            # LOW Priority - I think that this is deprecated?
            pass

    def _get_attribute_length(self, fld):
        if fld == 'DESOFLW':
            return 0 if self._DESOFLW is None else self._lengths['DESOFLOW']
        elif fld == 'DESITEM':
            return 0 if self._DESITEM is None else self._lengths['DESITEM']
        else:
            return super(DataExtensionHeader0, self)._get_attribute_length(fld)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'UserHeader':
            val = DESUserHeader.from_bytes(value, start)
            fields['UserHeader'] = val
            return start + val.get_bytes_length()
        elif attribute == 'DESTAG':
            val = value[start:start+cls._lengths['DESTAG']].decode('utf-8')
            fields['DESTAG'] = val
            if val.strip() not in ['TRE_OVERFLOW', 'Registered Extensions', 'Controlled Extensions']:
                fields['DESOFLW'] = None
                fields['DESITEM'] = None
            return start+cls._lengths['DESTAG']
        else:
            return super(DataExtensionHeader0, cls)._parse_attribute(fields, attribute, value, start)
