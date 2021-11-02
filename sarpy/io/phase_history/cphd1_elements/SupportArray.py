"""
The Support Array parameters definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from xml.etree import ElementTree

from typing import Union, List

from sarpy.io.xml.base import Serializable, ParametersCollection, get_node_value
from sarpy.io.xml.descriptors import FloatDescriptor, StringDescriptor, StringEnumDescriptor, \
    ParametersDescriptor, SerializableListDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .utils import homogeneous_dtype


class SupportArrayCore(Serializable):
    """
    The support array base case.
    """

    _fields = ('Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS', 'NODATA')
    _required = ('Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS')
    _numeric_format = {'X0': FLOAT_FORMAT, 'Y0': FLOAT_FORMAT, 'XSS': FLOAT_FORMAT, 'YSS': FLOAT_FORMAT}
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='The support array identifier.')  # type: str
    ElementFormat = StringDescriptor(
        'ElementFormat', _required, strict=DEFAULT_STRICT,
        docstring='The data element format.')  # type: str
    X0 = FloatDescriptor(
        'X0', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float
    Y0 = FloatDescriptor(
        'Y0', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float
    XSS = FloatDescriptor(
        'XSS', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    YSS = FloatDescriptor(
        'YSS', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float

    def __init__(self, Identifier=None, ElementFormat=None, X0=None, Y0=None,
                 XSS=None, YSS=None, NODATA=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        ElementFormat : str
        X0 : float
        Y0 : float
        XSS : float
        YSS : float
        NODATA : None|str
        kwargs
        """

        self._NODATA = None
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.ElementFormat = ElementFormat
        self.X0 = X0
        self.Y0 = Y0
        self.XSS = XSS
        self.YSS = YSS
        self.NODATA = NODATA
        super(SupportArrayCore, self).__init__(**kwargs)

    @property
    def NODATA(self):
        """
        None|str: The no data hex string value.
        """

        return self._NODATA

    @NODATA.setter
    def NODATA(self, value):
        if value is None:
            self._NODATA = None
            return

        if isinstance(value, ElementTree.Element):
            value = get_node_value(value)

        if isinstance(value, str):
            self._NODATA = value
        elif isinstance(value, bytes):
            self._NODATA = value.decode('utf-8')
        elif isinstance(value, int):
            raise NotImplementedError
        elif isinstance(value, float):
            raise NotImplementedError
        else:
            raise TypeError('Got unexpected type {}'.format(type(value)))

    def get_nodata_as_int(self):
        """
        Get the no data value as an integer value.

        Returns
        -------
        None|int
        """

        if self._NODATA is None:
            return None

        raise NotImplementedError

    def get_nodata_as_float(self):
        """
        Gets the no data value as a floating point value.

        Returns
        -------
        None|float
        """

        if self._NODATA is None:
            return None

        raise NotImplementedError

    def get_numpy_format(self):
        """
        Convert the element format to a numpy dtype (including endianness) and depth.

        Returns
        -------
        numpy.dtype, int
        """

        return homogeneous_dtype(self.ElementFormat, return_length=True)


class IAZArrayType(SupportArrayCore):
    """
    Array of scene surface heights expressed in image coordinate IAZ values (meters).
    Grid coordinates are image area coordinates (IAX, IAY).
    """

    _fields = ('Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS', 'NODATA')
    _required = ('Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS')
    # descriptors
    ElementFormat = StringEnumDescriptor(
        'ElementFormat', ('IAZ=F4;', ), _required, strict=DEFAULT_STRICT, default_value='IAZ=F4;',
        docstring='The data element format.')  # type: str

    def __init__(self, Identifier=None, ElementFormat='IAZ=F4;', X0=None, Y0=None, XSS=None, YSS=None,
                 NODATA=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        ElementFormat : str
        X0 : float
        Y0 : float
        XSS : float
        YSS : float
        NODATA : str
        kwargs
        """

        super(IAZArrayType, self).__init__(
            Identifier=Identifier, ElementFormat=ElementFormat, X0=X0, Y0=Y0,
            XSS=XSS, YSS=YSS, NODATA=NODATA, **kwargs)


class AntGainPhaseType(SupportArrayCore):
    """
    Antenna array with values are antenna gain and phase expressed in dB and
    cycles. Array coordinates are direction cosines with respect to the
    ACF (DCX, DCY).
    """

    _fields = ('Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS', 'NODATA')
    _required = ('Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS')
    # descriptors
    ElementFormat = StringEnumDescriptor(
        'ElementFormat', ('Gain=F4;Phase=F4;', ), _required, strict=DEFAULT_STRICT, default_value='Gain=F4;Phase=F4;',
        docstring='The data element format.')  # type: str

    def __init__(self, Identifier=None, ElementFormat='Gain=F4;Phase=F4;', X0=None,
                 Y0=None, XSS=None, YSS=None, NODATA=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        X0 : float
        Y0 : float
        XSS : float
        YSS : float
        NODATA : str
        kwargs
        """

        super(AntGainPhaseType, self).__init__(
            Identifier=Identifier, ElementFormat=ElementFormat, X0=X0, Y0=Y0,
            XSS=XSS, YSS=YSS, NODATA=NODATA, **kwargs)


class AddedSupportArrayType(SupportArrayCore):
    """
    Additional arrays (two-dimensional), where the content and format and units
    of each element are user defined.
    """

    _fields = (
        'Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS', 'NODATA',
        'XUnits', 'YUnits', 'ZUnits', 'Parameters')
    _required = (
        'Identifier', 'ElementFormat', 'X0', 'Y0', 'XSS', 'YSS',
        'XUnits', 'YUnits', 'ZUnits')
    _collections_tags = {
        'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    XUnits = StringDescriptor(
        'XUnits', _required, strict=DEFAULT_STRICT,
        docstring='The X units.')  # type: str
    YUnits = StringDescriptor(
        'YUnits', _required, strict=DEFAULT_STRICT,
        docstring='The Y units.')  # type: str
    ZUnits = StringDescriptor(
        'ZUnits', _required, strict=DEFAULT_STRICT,
        docstring='The Z units.')  # type: str
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Other necessary free-form parameters.')  # type: Union[None, ParametersCollection]

    def __init__(self, Identifier=None, ElementFormat=None,
                 X0=None, Y0=None, XSS=None, YSS=None, NODATA=None,
                 XUnits=None, YUnits=None, ZUnits=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        ElementFormat : str
        X0 : float
        Y0 : float
        XSS : float
        YSS : float
        NODATA : str
        XUnits : str
        YUnits : str
        ZUnits : str
        Parameters : ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.XUnits = XUnits
        self.YUnits = YUnits
        self.ZUnits = ZUnits
        self.Parameters = Parameters
        super(AddedSupportArrayType, self).__init__(
            Identifier=Identifier, ElementFormat=ElementFormat, X0=X0, Y0=Y0,
            XSS=XSS, YSS=YSS, NODATA=NODATA, **kwargs)


class SupportArrayType(Serializable):
    """
    Parameters that describe the binary support array(s) content and
    grid coordinates.
    """

    _fields = ('IAZArray', 'AntGainPhase', 'AddedSupportArray')
    _required = ()
    _collections_tags = {
        'IAZArray': {'array': False, 'child_tag': 'IAZArray'},
        'AntGainPhase': {'array': False, 'child_tag': 'AntGainPhase'},
        'AddedSupportArray': {'array': False, 'child_tag': 'AddedSupportArray'}}
    # descriptors
    IAZArray = SerializableListDescriptor(
        'IAZArray', IAZArrayType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Arrays of scene surface heights expressed in image coordinate IAZ '
                  'values (meters). Grid coordinates are image area coordinates '
                  '(IAX, IAY).')  # type: Union[None, List[IAZArrayType]]
    AntGainPhase = SerializableListDescriptor(
        'AntGainPhase', AntGainPhaseType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Antenna arrays with values are antenna gain and phase expressed in dB '
                  'and cycles. Array coordinates are direction cosines with respect to '
                  'the ACF (DCX, DCY).')  # type: Union[None, List[AntGainPhaseType]]
    AddedSupportArray = SerializableListDescriptor(
        'AddedSupportArray', AddedSupportArrayType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional arrays (two-dimensional), where the content and format and units of each '
                  'element are user defined.')  # type: Union[None, List[AddedSupportArrayType]]

    def __init__(self, IAZArray=None, AntGainPhase=None, AddedSupportArray=None, **kwargs):
        """

        Parameters
        ----------
        IAZArray : None|List[IAZArrayType]
        AntGainPhase : None|List[AntGainPhaseType]
        AddedSupportArray : None|List[AddedSupportArrayType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.IAZArray = IAZArray
        self.AntGainPhase = AntGainPhase
        self.AddedSupportArray = AddedSupportArray
        super(SupportArrayType, self).__init__(**kwargs)

    def find_support_array(self, identifier):
        """
        Find and return the details for support array associated with the given identifier.

        Parameters
        ----------
        identifier : str

        Returns
        -------
        IAZArrayType|AntGainPhaseType|AddedSupportArrayType
        """

        if self.IAZArray is not None:
            for entry in self.IAZArray:
                if entry.Identifier == identifier:
                    return entry

        if self.AntGainPhase is not None:
            for entry in self.AntGainPhase:
                if entry.Identifier == identifier:
                    return entry

        if self.AddedSupportArray is not None:
            for entry in self.AddedSupportArray:
                if entry.Identifier == identifier:
                    return entry

        raise KeyError('Identifier {} not associated with a support array.'.format(identifier))
