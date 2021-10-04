"""
The Compensated Received Signal Data 1.0 definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")

from xml.etree import ElementTree
from collections import OrderedDict
from typing import Union
import numpy

from sarpy.io.xml.base import Serializable, find_children
from sarpy.io.xml.descriptors import SerializableDescriptor, IntegerDescriptor, \
    StringDescriptor

from .base import DEFAULT_STRICT
from .CollectionID import CollectionIDType
from .Global import GlobalType
from sarpy.io.phase_history.cphd1_elements.SceneCoordinates import SceneCoordinatesType
from .Data import DataType
from .Channel import ChannelType
from .PVP import PVPType
from sarpy.io.phase_history.cphd1_elements.SupportArray import SupportArrayType
from sarpy.io.phase_history.cphd1_elements.Dwell import DwellType
from .ReferenceGeometry import ReferenceGeometryType
from .Antenna import AntennaType
from .ErrorParameters import ErrorParametersType
from sarpy.io.phase_history.cphd1_elements.ProductInfo import ProductInfoType
from sarpy.io.phase_history.cphd1_elements.GeoInfo import GeoInfoType
from sarpy.io.complex.sicd_elements.MatchInfo import MatchInfoType


#########
# Module variables
_CRSD_SPECIFICATION_VERSION = '1.0.0'
_CRSD_SPECIFICATION_DATE = '2021-06-30T00:00:00Z'
_CRSD_SPECIFICATION_NAMESPACE = 'http://api.nsgreg.nga.mil/schema/crsd/1.0.0'
_CRSD_SECTION_TERMINATOR = b'\f\n'


#########
# CRSD header object

def _parse_crsd_header_field(line):
    """
    Parse the CRSD header field, or return `None` as a termination signal.

    Parameters
    ----------
    line : bytes

    Returns
    -------
    None|(str, str)
    """

    if line.startswith(_CRSD_SECTION_TERMINATOR):
        return None
    parts = line.split(b' := ')
    if len(parts) != 2:
        raise ValueError('Cannot extract CRSD header value from line {}'.format(line))
    fld = parts[0].strip().decode('utf-8')
    val = parts[1].strip().decode('utf-8')
    return fld, val


class CRSDHeaderBase(object):
    _fields = ()
    _required = ()

    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_file_object(cls, fi):
        """
        Extract the CRSD header object from a file opened in byte mode.
        This file object is assumed to be at the correct location for the
        CRSD header.

        Parameters
        ----------
        fi
            The open file object, which will be progressively read.

        Returns
        -------
        CRSDHeaderBase
        """

        the_dict = {}
        while True:
            line = fi.readline()
            res = _parse_crsd_header_field(line)
            if res is None:
                break
            else:
                fld, val = res
            if fld not in cls._fields:
                raise ValueError('Cannot extract CRSD header value from line {}'.format(line))
            the_dict[fld] = val
        return cls(**the_dict)


class CRSDHeader(CRSDHeaderBase):
    _fields = (
        'XML_BLOCK_SIZE', 'XML_BLOCK_BYTE_OFFSET', 'SUPPORT_BLOCK_SIZE', 'SUPPORT_BLOCK_BYTE_OFFSET',
        'PVP_BLOCK_SIZE', 'PVP_BLOCK_BYTE_OFFSET', 'SIGNAL_BLOCK_SIZE', 'SIGNAL_BLOCK_BYTE_OFFSET',
        'CLASSIFICATION', 'RELEASE_INFO')
    _required = (
        'XML_BLOCK_SIZE', 'XML_BLOCK_BYTE_OFFSET', 'PVP_BLOCK_SIZE', 'PVP_BLOCK_BYTE_OFFSET',
        'SIGNAL_BLOCK_SIZE', 'SIGNAL_BLOCK_BYTE_OFFSET', 'CLASSIFICATION', 'RELEASE_INFO')
    # descriptor
    XML_BLOCK_SIZE = IntegerDescriptor(
        'XML_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the XML instance that describes the product in bytes. '
                  'Size does NOT include the 2 bytes of the section terminator.')  # type: int
    XML_BLOCK_BYTE_OFFSET = IntegerDescriptor(
        'XML_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the XML block in bytes.')  # type: int
    SUPPORT_BLOCK_SIZE = IntegerDescriptor(
        'SUPPORT_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the Support block in bytes. Note - If the Support block is omitted, this '
                  'is not included.')  # type: int
    SUPPORT_BLOCK_BYTE_OFFSET = IntegerDescriptor(
        'SUPPORT_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the Support block in bytes. Note - If the Support '
                  'block is omitted, this is not included.')  # type: int
    PVP_BLOCK_SIZE = IntegerDescriptor(
        'PVP_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the PVP block in bytes.')  # type: int
    PVP_BLOCK_BYTE_OFFSET = IntegerDescriptor(
        'PVP_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the PVP block in bytes.')  # type: int
    SIGNAL_BLOCK_SIZE = IntegerDescriptor(
        'SIGNAL_BLOCK_SIZE', _required, strict=True,
        docstring='Size of the Signal block in bytes.')  # type: int
    SIGNAL_BLOCK_BYTE_OFFSET = IntegerDescriptor(
        'SIGNAL_BLOCK_BYTE_OFFSET', _required, strict=True,
        docstring='Offset to the first byte of the Signal block in bytes.')  # type: int
    CLASSIFICATION = StringDescriptor(
        'CLASSIFICATION', _required, strict=True, default_value='UNCLASSIFIED',
        docstring='Product classification information that is human-readable.')  # type: str
    RELEASE_INFO = StringDescriptor(
        'RELEASE_INFO', _required, strict=True, default_value='UNRESTRICTED',
        docstring='Product release information that is human-readable.')  # type: str

    def __init__(self, XML_BLOCK_SIZE=None, XML_BLOCK_BYTE_OFFSET=None,
                 SUPPORT_BLOCK_SIZE=None, SUPPORT_BLOCK_BYTE_OFFSET=None,
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
        super(CRSDHeader, self).__init__()

    def to_string(self):
        """
        Forms a CRSD file header string (not including the section terminator) from populated attributes.
        """
        return ('CRSD/{}\n'.format(_CRSD_SPECIFICATION_VERSION)
                + ''.join(["{} := {}\n".format(f, getattr(self, f))
                           for f in self._fields if getattr(self, f) is not None]))


class CRSDType(Serializable):
    """
    The Compensated Received Signal Data definition.
    """

    _fields = (
        'CollectionID', 'Global', 'SceneCoordinates', 'Data', 'Channel', 'PVP',
        'SupportArray', 'Dwell', 'ReferenceGeometry', 'Antenna',
        'ErrorParameters', 'ProductInfo', 'GeoInfo', 'MatchInfo')
    _required = (
        'CollectionID', 'Global', 'Data', 'Channel', 'PVP', 'ReferenceGeometry')
    _collections_tags = {'GeoInfo': {'array': 'False', 'child_tag': 'GeoInfo'}}
    # descriptors
    CollectionID = SerializableDescriptor(
        'CollectionID', CollectionIDType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the collection.')  # type: CollectionIDType
    Global = SerializableDescriptor(
        'Global', GlobalType, _required, strict=DEFAULT_STRICT,
        docstring='Global parameters that apply to metadata components and CRSD '
                  'signal arrays.')  # type: GlobalType
    SceneCoordinates = SerializableDescriptor(
        'SceneCoordinates', SceneCoordinatesType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that define geographic coordinates of the imaged '
                  'scene.')  # type: Union[None, SceneCoordinatesType]
    Data = SerializableDescriptor(
        'Data', DataType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe binary data components contained in '
                  'the product.')  # type: DataType
    Channel = SerializableDescriptor(
        'Channel', ChannelType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the data channels contained in the '
                  'product.')  # type: ChannelType
    PVP = SerializableDescriptor(
        'PVP', PVPType, _required, strict=DEFAULT_STRICT,
        docstring='Structure used to specify the Per Vector Parameters provided for '
                  'each channel of a given product.')  # type: PVPType
    SupportArray = SerializableDescriptor(
        'SupportArray', SupportArrayType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the binary support array(s) content and '
                  'grid coordinates.')  # type: Union[None, SupportArrayType]
    Dwell = SerializableDescriptor(
        'Dwell', DwellType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that specify the SAR dwell time supported by the signal '
                  'arrays contained in the CRSD product.')  # type: Union[None, DwellType]
    ReferenceGeometry = SerializableDescriptor(
        'ReferenceGeometry', ReferenceGeometryType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the collection geometry for the reference '
                  'vector of the reference channel.')  # type: ReferenceGeometryType
    Antenna = SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the antennas antennas used '
                  'to collect the signal array(s).')  # type: Union[None, AntennaType]
    ErrorParameters = SerializableDescriptor(
        'ErrorParameters', ErrorParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the statistics of errors in measured or estimated parameters '
                  'that describe the collection.')  # type: Union[None, ErrorParametersType]
    ProductInfo = SerializableDescriptor(
        'ProductInfo', ProductInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that provide general information about the CRSD product '
                  'and/or the derived products that may be created '
                  'from it.')  # type: Union[None, ProductInfoType]
    MatchInfo = SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Information about other collections that are matched to the collection from which '
                  'this CRSD product was generated.')  # type: Union[None, MatchInfoType]

    def __init__(self, CollectionID=None, Global=None, SceneCoordinates=None, Data=None,
                 Channel=None, PVP=None, SupportArray=None, Dwell=None, ReferenceGeometry=None,
                 Antenna=None, ErrorParameters=None, ProductInfo=None,
                 GeoInfo=None, MatchInfo=None, **kwargs):
        """

        Parameters
        ----------
        CollectionID : CollectionIDType
        Global : GlobalType
        SceneCoordinates : None|SceneCoordinatesType
        Data : DataType
        Channel : ChannelType
        PVP : PVPType
        SupportArray : None|SupportArrayType
        Dwell : None|DwellType
        ReferenceGeometry : ReferenceGeometryType
        Antenna : None|AntennaType
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

        super(CRSDType, self).__init__(**kwargs)

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
        kwargs['GeoInfo'] = find_children(node, 'GeoInfo', xml_ns, gi_key)
        return super(CRSDType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(CRSDType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity,
            strict=strict, exclude=exclude+('GeoInfo', ))
        # slap on the GeoInfo children
        if self._GeoInfo is not None and len(self._GeoInfo) > 0:
            for entry in self._GeoInfo:
                entry.to_node(doc, 'GeoInfo', ns_key=ns_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(CRSDType, self).to_dict(
            check_validity=check_validity, strict=strict, exclude=exclude+('GeoInfo', ))
        # slap on the GeoInfo children
        if len(self.GeoInfo) > 0:
            out['GeoInfo'] = [entry.to_dict(
                check_validity=check_validity, strict=strict) for entry in self._GeoInfo]
        return out

    def to_xml_bytes(self, urn=None, tag='CRSD', check_validity=False, strict=DEFAULT_STRICT):
        if urn is None:
            urn = _CRSD_SPECIFICATION_NAMESPACE
        return super(CRSDType, self).to_xml_bytes(
            urn=urn, tag=tag, check_validity=check_validity, strict=strict)

    def to_xml_string(self, urn=None, tag='CRSD', check_validity=False, strict=DEFAULT_STRICT):
        return self.to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict).decode('utf-8')

    def make_file_header(self, xml_offset=1024):
        """
        Forms a CRSD file header consistent with the information in the Data and CollectionID nodes.

        Parameters
        ----------
        xml_offset : int, optional
            Offset in bytes to the first byte of the XML block. If the provided value
            is not large enough to account for the length of the file header
            string, a larger value is chosen.

        Returns
        -------
        header : sarpy.io.phase_history.cphd1_elements.CRSD.CRSDHeader
        """

        _kvps = OrderedDict()

        def _align(val):
            align_to = 64
            return int(numpy.ceil(float(val)/align_to)*align_to)

        _kvps['XML_BLOCK_SIZE'] = len(self.to_xml_string())
        _kvps['XML_BLOCK_BYTE_OFFSET'] = xml_offset
        block_end = _kvps['XML_BLOCK_BYTE_OFFSET'] + _kvps['XML_BLOCK_SIZE'] + len(_CRSD_SECTION_TERMINATOR)

        if self.Data.NumSupportArrays > 0:
            _kvps['SUPPORT_BLOCK_SIZE'] = self.Data.calculate_support_block_size()
            _kvps['SUPPORT_BLOCK_BYTE_OFFSET'] = _align(block_end)
            block_end = _kvps['SUPPORT_BLOCK_BYTE_OFFSET'] + _kvps['SUPPORT_BLOCK_SIZE']

        _kvps['PVP_BLOCK_SIZE'] = self.Data.calculate_pvp_block_size()
        _kvps['PVP_BLOCK_BYTE_OFFSET'] = _align(block_end)
        block_end = _kvps['PVP_BLOCK_BYTE_OFFSET'] + _kvps['PVP_BLOCK_SIZE']

        _kvps['SIGNAL_BLOCK_SIZE'] = self.Data.calculate_signal_block_size()
        _kvps['SIGNAL_BLOCK_BYTE_OFFSET'] = _align(block_end)
        _kvps['CLASSIFICATION'] = self.CollectionID.Classification
        _kvps['RELEASE_INFO'] = self.CollectionID.ReleaseInfo

        header = CRSDHeader(**_kvps)
        header_str = header.to_string()
        min_xml_offset = len(header_str) + len(_CRSD_SECTION_TERMINATOR)
        if _kvps['XML_BLOCK_BYTE_OFFSET'] < min_xml_offset:
            header = self.make_file_header(xml_offset=_align(min_xml_offset + 32))

        return header

    def get_pvp_dtype(self):
        """
        Gets the dtype for the corresponding PVP structured array. Note that they
        must all have homogeneous dtype.

        Returns
        -------
        numpy.dtype
            This will be a compound dtype for a structured array.
        """

        if self.PVP is None:
            raise ValueError('No PVP defined.')
        return self.PVP.get_vector_dtype()
