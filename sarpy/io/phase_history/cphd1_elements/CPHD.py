# -*- coding: utf-8 -*-
"""
The Compensated Phase History Data 1.0.1 definition.
"""

from xml.etree import ElementTree
from collections import OrderedDict
from typing import Union

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _SerializableDescriptor, \
    _IntegerDescriptor, _StringDescriptor, _find_children

from .CollectionID import CollectionIDType
from .Global import GlobalType
from .SceneCoordinates import SceneCoordinatesType
from .Data import DataType
from .Channel import ChannelType
from .PVP import PVPType
from .SupportArray import SupportArrayType
from .Dwell import DwellType
from .ReferenceGeometry import ReferenceGeometryType
from .Antenna import AntennaType
from .TxRcv import TxRcvType
from .ErrorParameters import ErrorParametersType
from .ProductInfo import ProductInfoType
from .GeoInfo import GeoInfoType
from sarpy.io.complex.sicd_elements.MatchInfo import MatchInfoType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#########
# Module variables
_CPHD_SPECIFICATION_VERSION = '1.0'
_CPHD_SPECIFICATION_DATE = '2018-05-21T00:00:00Z'
_CPHD_SPECIFICATION_NAMESPACE = 'urn:CPHD:1.0.1'


#########
# CPHD header object

def _parse_cphd_header_field(line):
    """
    Parse the CPHD header field, or return `None` as a termination signal.

    Parameters
    ----------
    line : bytes

    Returns
    -------
    None|(str, str)
    """

    if line.startswith(b'\f\n'):
        return None
    parts = line.strip().split(b':=')
    if len(parts) != 2:
        raise ValueError('Cannot extract CPHD header value from line {}'.format(line))
    fld = parts[0].strip().decode('utf-8')
    val = parts[1].strip().decode('utf-8')
    return fld, val


class CPHDHeaderBase(object):
    _fields = ()
    _required = ()

    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_file_object(cls, fi):
        """
        Extract the CPHD header object from a file opened in byte mode.
        This file object is assumed to be at the correct location for the
        CPHD header.

        Parameters
        ----------
        fi
            The open file object, which will be progressively read.

        Returns
        -------
        CPHDHeaderBase
        """

        the_dict = {}
        while True:
            line = fi.readline()
            res = _parse_cphd_header_field(line)
            if res is None:
                break
            else:
                fld, val = res
            if fld not in cls._fields:
                raise ValueError('Cannot extract CPHD header value from line {}'.format(line))
            the_dict[fld] = val
        return cls(**the_dict)


class CPHDHeader(CPHDHeaderBase):
    _fields = (
        'XML_BLOCK_SIZE', 'XML_BLOCK_BYTE_OFFSET', 'SUPPORT_BLOCK_SIZE', 'SUPPORT_BLOCK_BYTE_OFFSET',
        'PVP_BLOCK_SIZE', 'PVP_BLOCK_BYTE_OFFSET', 'SIGNAL_BLOCK_SIZE', 'SIGNAL_BLOCK_BYTE_OFFSET',
        'CLASSIFICATION', 'RELEASE_INFO')
    _required = (
        'XML_BLOCK_SIZE', 'XML_BLOCK_BYTE_OFFSET', 'PVP_BLOCK_SIZE', 'PVP_BLOCK_BYTE_OFFSET',
        'SIGNAL_BLOCK_SIZE', 'SIGNAL_BLOCK_BYTE_OFFSET', 'CLASSIFICATION', 'RELEASE_INFO')
    # descriptor
    XML_BLOCK_SIZE = _IntegerDescriptor(
        'XML_BLOCK_SIZE', _required, strict=True,
        docstring='ize of the XML instance that describes the product in bytes. '
                  'Size does NOT include the 2 bytes of the section terminator.')  # type: int
    XML_BLOCK_BYTE_OFFSET = _IntegerDescriptor(
        'XML_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the XML block in bytes.')  # type: int
    SUPPORT_BLOCK_SIZE = _IntegerDescriptor(
        'SUPPORT_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the Support block in bytes. Note - If the Support block is omitted, this '
                  'is not included.')  # type: int
    SUPPORT_BLOCK_BYTE_OFFSET = _IntegerDescriptor(
        'SUPPORT_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the Support block in bytes. Note - If the Support '
                  'block is omitted, this is not included.')  # type: int
    PVP_BLOCK_SIZE = _IntegerDescriptor(
        'PVP_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the PVP block in bytes.')  # type: int
    PVP_BLOCK_BYTE_OFFSET = _IntegerDescriptor(
        'PVP_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the PVP block in bytes.')  # type: int
    SIGNAL_BLOCK_SIZE = _IntegerDescriptor(
        'SIGNAL_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the Signal block in bytes.')  # type: int
    SIGNAL_BLOCK_BYTE_OFFSET = _IntegerDescriptor(
        'SIGNAL_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the Signal block in bytes.')  # type: int
    CLASSIFICATION = _StringDescriptor(
        'CLASSIFICATION', _required, strict=True, default_value='UNCLASSIFIED',
        docstring='Product classification information that is human-readable.')  # type: str
    RELEASE_INFO = _StringDescriptor(
        'RELEASE_INFO', _required, strict=True, default_value='UNRESTRICTED',
        docstring='Product release information that is human-readable.')  # type: str

    def __init__(self, XML_BLOCK_SIZE=None, XML_BLOCK_BYTE_OFFSET=None,
                 SUPPORT_BLOCK_SIZE= None, SUPPORT_BLOCK_BYTE_OFFSET=None,
                 PVP_BLOCK_SIZE=None, PVP_BLOCK_BYTE_OFFSET=None,
                 SIGNAL_BLOCK_SIZE=None, SIGNAL_BLOCK_BYTE_OFFSET=None,
                 CLASSIFICATION='UNCLASSIFIED', RELEASE_INFO='UNRESTRICTED'):
        self.XML_BLOCK_SIZE = XML_BLOCK_SIZE
        self.XML_BLOCK_BYTE_OFFSET = XML_BLOCK_BYTE_OFFSET
        self.SUPPORT_BLOCK_SIZE = SUPPORT_BLOCK_SIZE
        self.SUPPORT_BLOCK_BYTE_OFFSET = SUPPORT_BLOCK_BYTE_OFFSET
        self.PVP_BLOCK_SIZE = PVP_BLOCK_SIZE
        self.PVP_BLOCK_BYTE_OFFSET = PVP_BLOCK_BYTE_OFFSET
        self.SIGNAL_BLOCK_SIZE = SIGNAL_BLOCK_SIZE
        self.SIGNAL_BLOCK_BYTE_OFFSET = SIGNAL_BLOCK_BYTE_OFFSET
        self.CLASSIFICATION = CLASSIFICATION
        self.RELEASE_INFO = RELEASE_INFO
        super(CPHDHeader, self).__init__()


class CPHDType(Serializable):
    """
    The Compensated Phase History Data definition.
    """

    _fields = (
        'CollectionID', 'Global', 'SceneCoordinates', 'Data', 'Channel', 'PVP',
        'SupportArray', 'Dwell', 'ReferenceGeometry', 'Antenna', 'TxRcv',
        'ErrorParameters', 'ProductInfo', 'GeoInfo', 'MatchInfo')
    _required = (
        'CollectionID', 'Global', 'SceneCoordinates', 'Data', 'Channel', 'PVP',
        'Dwell', 'ReferenceGeometry')
    _collections_tags = {'GeoInfo': {'array': 'False', 'child_tag': 'GeoInfo'}}
    # descriptors
    CollectionID = _SerializableDescriptor(
        'CollectionID', CollectionIDType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the collection.')  # type: CollectionIDType
    Global = _SerializableDescriptor(
        'Global', GlobalType, _required, strict=DEFAULT_STRICT,
        docstring='Global parameters that apply to metadata components and CPHD '
                  'signal arrays.')  # type: GlobalType
    SceneCoordinates = _SerializableDescriptor(
        'SceneCoordinates', SceneCoordinatesType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that define geographic coordinates for in the imaged '
                  'scene.')  # type: SceneCoordinatesType
    Data = _SerializableDescriptor(
        'Data', DataType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe binary data components contained in '
                  'the product.')  # type: DataType
    Channel = _SerializableDescriptor(
        'Channel', ChannelType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the data channels contained in the '
                  'product.')  # type: ChannelType
    PVP = _SerializableDescriptor(
        'PVP', PVPType, _required, strict=DEFAULT_STRICT,
        docstring='Structure specifying the Per Vector parameters provided for '
                  'each channel of a given product.')  # type: PVPType
    SupportArray = _SerializableDescriptor(
        'SupportArray', SupportArrayType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the binary support array(s) content and '
                  'grid coordinates.')  # type: Union[None, SupportArrayType]
    Dwell = _SerializableDescriptor(
        'Dwell', DwellType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that specify the dwell time supported by the signal '
                  'arrays contained in the CPHD product.')  # type: DwellType
    ReferenceGeometry = _SerializableDescriptor(
        'ReferenceGeometry', ReferenceGeometryType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the collection geometry for the reference '
                  'vector of the reference channel.')  # type: ReferenceGeometryType
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the transmit and receive antennas used '
                  'to collect the signal array(s).')  # type: Union[None, AntennaType]
    TxRcv = _SerializableDescriptor(
        'TxRcv', TxRcvType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the transmitted waveform(s) and receiver configurations '
                  'used in the collection.')  # type: Union[None, TxRcvType]
    ErrorParameters = _SerializableDescriptor(
        'ErrorParameters', ErrorParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the statistics of errors in measured or estimated parameters '
                  'that describe the collection.')  # type: Union[None, ErrorParametersType]
    ProductInfo = _SerializableDescriptor(
        'ProductInfo', ProductInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that provide general information about the CPHD product '
                  'and/or the derived products that may be created '
                  'from it.')  # type: Union[None, ProductInfoType]
    MatchInfo = _SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Information about other collections that are matched to the collection from which '
                  'this CPHD product was generated.')  # type: Union[None, MatchInfoType]

    def __init__(self, CollectionID=None, Global=None, SceneCoordinates=None, Data=None,
                 Channel=None, PVP=None, SupportArray=None, Dwell=None, ReferenceGeometry=None,
                 Antenna=None, TxRcv=None, ErrorParameters=None, ProductInfo=None,
                 GeoInfo=None, MatchInfo=None, **kwargs):
        """

        Parameters
        ----------
        CollectionID : CollectionIDType
        Global : GlobalType
        SceneCoordinates : SceneCoordinatesType
        Data : DataType
        Channel : ChannelType
        PVP : PVPType
        SupportArray : None|SupportArrayType
        Dwell : DwellType
        ReferenceGeometry : ReferenceGeometryType
        Antenna : None|AntennaType
        TxRcv : None|TxRcvType
        ErrorParameters : None|ErrorParametersType
        ProductInfo : None|ProductInfoType
        GeoInfo : None|List[GeoInfoType]|GeoInfoType
        MatchInfo : None|MatchInfoType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CollectionID = CollectionID
        self.Global = Global
        self.SceneCoordinates = SceneCoordinates
        self.Data = Data
        self.Channel = Channel
        self.PVP = PVP
        self.SupportArray = SupportArray
        self.Dwell = Dwell
        self.ReferenceGeometry = ReferenceGeometry
        self.Antenna = Antenna
        self.TxRcv = TxRcv
        self.ErrorParameters = ErrorParameters
        self.ProductInfo = ProductInfo
        self.MatchInfo = MatchInfo

        self._GeoInfo = []
        if GeoInfo is None:
            pass
        elif isinstance(GeoInfo, GeoInfoType):
            self.addGeoInfo(GeoInfo)
        elif isinstance(GeoInfo, (list, tuple)):
            for el in GeoInfo:
                self.addGeoInfo(el)
        else:
            raise ('GeoInfo got unexpected type {}'.format(type(GeoInfo)))

        super(CPHDType, self).__init__(**kwargs)

    @property
    def GeoInfo(self):
        """
        List[GeoInfoType]: Parameters that describe a geographic feature.
        """

        return self._GeoInfo

    def getGeoInfo(self, key):
        """
        Get GeoInfo(s) with name attribute == `key`.

        Parameters
        ----------
        key : str

        Returns
        -------
        List[GeoInfoType]
        """

        return [entry for entry in self._GeoInfo if entry.name == key]

    def addGeoInfo(self, value):
        """
        Add the given GeoInfo to the GeoInfo list.

        Parameters
        ----------
        value : GeoInfoType

        Returns
        -------
        None
        """

        if isinstance(value, ElementTree.Element):
            gi_key = self._child_xml_ns_key.get('GeoInfo', self._xml_ns_key)
            value = GeoInfoType.from_node(value, self._xml_ns, ns_key=gi_key)
        elif isinstance(value, dict):
            value = GeoInfoType.from_dict(value)

        if isinstance(value, GeoInfoType):
            self._GeoInfo.append(value)
        else:
            raise TypeError('Trying to set GeoInfo element with unexpected type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = OrderedDict()
        gi_key = cls._child_xml_ns_key.get('GeoInfo', ns_key)
        kwargs['GeoInfo'] = _find_children(node, 'GeoInfo', xml_ns, gi_key)
        return super(CPHDType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(CPHDType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity, strict=strict, exclude=exclude+('GeoInfo', ))
        # slap on the GeoInfo children
        if self._GeoInfo is not None and len(self._GeoInfo) > 0:
            for entry in self._GeoInfo:
                entry.to_node(doc, 'GeoInfo', ns_key=ns_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(CPHDType, self).to_dict(check_validity=check_validity, strict=strict, exclude=exclude+('GeoInfo', ))
        # slap on the GeoInfo children
        if len(self.GeoInfo) > 0:
            out['GeoInfo'] = [entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._GeoInfo]
        return out

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        return super(CPHDType, self).to_xml_bytes(
            urn=_CPHD_SPECIFICATION_NAMESPACE, tag=tag, check_validity=check_validity, strict=strict)
