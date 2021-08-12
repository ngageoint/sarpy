"""
Multipurpose basic SIDD elements
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from collections import OrderedDict
from typing import Union

import numpy

from sarpy.compliance import int_func
from sarpy.io.xml.base import Serializable, Arrayable, get_node_value, create_text_node, create_new_node, find_children
from sarpy.io.xml.descriptors import SerializableDescriptor, IntegerDescriptor, \
    FloatDescriptor, FloatModularDescriptor, StringDescriptor, StringEnumDescriptor

from .base import DEFAULT_STRICT
from sarpy.io.complex.sicd_elements.blocks import XYZType as XYZTypeBase, XYZPolyType as XYZPolyTypeBase, \
    LatLonType as LatLonTypeBase, LatLonCornerType as LatLonCornerTypeBase, \
    RowColType as RowColIntTypeBase, RowColArrayElement as RowColArrayElementBase, \
    Poly1DType as Poly1DTypeBase, Poly2DType as Poly2DTypeBase, \
    LatLonCornerStringType as LatLonCornerStringTypeBase, LatLonArrayElementType as LatLonArrayElementTypeBase
from sarpy.io.complex.sicd_elements.ErrorStatistics import ErrorStatisticsType as ErrorStatisticsTypeBase
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType as RadiometricTypeBase
from sarpy.io.complex.sicd_elements.MatchInfo import MatchInfoType as MatchInfoTypeBase
from sarpy.io.complex.sicd_elements.GeoData import GeoInfoType as GeoInfoTypeBase
from sarpy.io.complex.sicd_elements.CollectionInfo import RadarModeType as RadarModeTypeBase


############
# the SICommon namespace elements

class XYZType(XYZTypeBase):
    _child_xml_ns_key = {'X': 'sicommon', 'Y': 'sicommon', 'Z': 'sicommon'}


class LatLonType(LatLonTypeBase):
    _child_xml_ns_key = {'Lat': 'sicommon', 'Lon': 'sicommon'}


class LatLonCornerType(LatLonCornerTypeBase):
    _child_xml_ns_key = {'Lat': 'sicommon', 'Lon': 'sicommon'}


class LatLonCornerStringType(LatLonCornerStringTypeBase):
    _child_xml_ns_key = {'Lat': 'sicommon', 'Lon': 'sicommon'}


class LatLonArrayElementType(LatLonArrayElementTypeBase):
    _child_xml_ns_key = {'Lat': 'sicommon', 'Lon': 'sicommon'}


class RangeAzimuthType(Serializable, Arrayable):
    """
    Represents range and azimuth.
    """
    _fields = ('Range', 'Azimuth')
    _required = ('Range', 'Azimuth')
    _numeric_format = {key: '0.16G' for key in _fields}
    _child_xml_ns_key = {'Range': 'sicommon', 'Azimuth': 'sicommon'}
    # Descriptor
    Range = FloatDescriptor(
        'Range', _required, strict=DEFAULT_STRICT,
        docstring='The range in meters.')  # type: float
    Azimuth = FloatDescriptor(
        'Azimuth', _required, strict=DEFAULT_STRICT,
        docstring='The azimuth in degrees.')  # type: float

    def __init__(self, Range=None, Azimuth=None, **kwargs):
        """

        Parameters
        ----------
        Range : float
        Azimuth : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Range = Range
        self.Azimuth = Azimuth
        super(RangeAzimuthType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Range, Azimuth]
        """

        return numpy.array([self.Range, self.Azimuth], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Range, Azimuth]

        Returns
        -------
        RangeAzimuthType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Range=array[0], Azimuth=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class AngleMagnitudeType(Serializable, Arrayable):
    """
    Represents a magnitude and angle.
    """

    _fields = ('Angle', 'Magnitude')
    _required = ('Angle', 'Magnitude')
    _numeric_format = {key: '0.16G' for key in _fields}
    _child_xml_ns_key = {'Angle': 'sicommon', 'Magnitude': 'sicommon'}
    # Descriptor
    Angle = FloatModularDescriptor(
        'Angle', 180.0, _required, strict=DEFAULT_STRICT,
        docstring='The angle.')  # type: float
    Magnitude = FloatDescriptor(
        'Magnitude', _required, strict=DEFAULT_STRICT, bounds=(0.0, None),
        docstring='The magnitude.')  # type: float

    def __init__(self, Angle=None, Magnitude=None, **kwargs):
        """

        Parameters
        ----------
        Angle : float
        Magnitude : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Angle = Angle
        self.Magnitude = Magnitude
        super(AngleMagnitudeType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Angle, Magnitude]
        """

        return numpy.array([self.Angle, self.Magnitude], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Angle, Magnitude]

        Returns
        -------
        AngleMagnitudeType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Angle=array[0], Magnitude=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class RowColIntType(RowColIntTypeBase):
    _child_xml_ns_key = {'Row': 'sicommon', 'Col': 'sicommon'}


class RowColArrayElement(RowColArrayElementBase):
    _child_xml_ns_key = {'Row': 'sicommon', 'Col': 'sicommon'}


class RowColDoubleType(Serializable, Arrayable):
    _fields = ('Row', 'Col')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    _child_xml_ns_key = {'Row': 'sicommon', 'Col': 'sicommon'}
    # Descriptors
    Row = FloatDescriptor(
        'Row', _required, strict=True, docstring='The Row attribute.')  # type: float
    Col = FloatDescriptor(
        'Col', _required, strict=True, docstring='The Column attribute.')  # type: float

    def __init__(self, Row=None, Col=None, **kwargs):
        """
        Parameters
        ----------
        Row : float
        Col : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Row, self.Col = Row, Col
        super(RowColDoubleType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Row, Col]
        """

        return numpy.array([self.Row, self.Col], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Row, Col]

        Returns
        -------
        RowColDoubleType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Row=array[0], Col=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class Poly1DType(Poly1DTypeBase):
    _child_xml_ns_key = {'Coefs': 'sicommon'}


class Poly2DType(Poly2DTypeBase):
    _child_xml_ns_key = {'Coefs': 'sicommon'}


class XYZPolyType(XYZPolyTypeBase):
    _child_xml_ns_key = {'X': 'sicommon', 'Y': 'sicommon', 'Z': 'sicommon'}


class ErrorStatisticsType(ErrorStatisticsTypeBase):
    _child_xml_ns_key = {'CompositeSCP': 'sicommon', 'Components': 'sicommon', 'AdditionalParms': 'sicommon'}


class RadiometricType(RadiometricTypeBase):
    _child_xml_ns_key = {
        'NoiseLevel': 'sicommon', 'RCSSFPoly': 'sicommon', 'SigmaZeroSFPoly': 'sicommon',
        'BetaZeroSFPoly': 'sicommon', 'GammaZeroSFPoly': 'sicommon'}


class MatchInfoType(MatchInfoTypeBase):
    _child_xml_ns_key = {'NumMatchTypes': 'sicommon', 'MatchTypes': 'sicommon'}


class GeoInfoType(GeoInfoTypeBase):
    _child_xml_ns_key = {
        'Descriptions': 'sicommon', 'Point': 'sicommon', 'Line': 'sicommon',
        'Polygon': 'sicommon'}


class RadarModeType(RadarModeTypeBase):
    _child_xml_ns_key = {'ModeType': 'sicommon', 'ModeID': 'sicommon'}


class ReferencePointType(Serializable):
    """
    A reference point.
    """

    _fields = ('ECEF', 'Point', 'name')
    _required = ('ECEF', 'Point')
    _set_as_attribute = ('name', )
    _child_xml_ns_key = {'ECEF': 'sicommon', 'Point': 'sicommon'}
    # Descriptor
    ECEF = SerializableDescriptor(
        'ECEF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The ECEF coordinates of the reference point.')  # type: XYZType
    Point = SerializableDescriptor(
        'Point', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='The pixel coordinates of the reference point.')  # type: RowColDoubleType
    name = StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='Used for implementation specific signifier for the reference point.')  # type: Union[None, str]

    def __init__(self, ECEF=None, Point=None, name=None, **kwargs):
        """

        Parameters
        ----------
        ECEF : XYZType|numpy.ndarray|list|tuple
        Point : RowColDoubleType|numpy.ndarray|list|tuple
        name : None|str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ECEF = ECEF
        self.Point = Point
        self.name = name
        super(ReferencePointType, self).__init__(**kwargs)

# The end of the SICommon namespace
#####################


#################
# Filter Type

class PredefinedFilterType(Serializable):
    """
    The predefined filter type.
    """
    _fields = ('DatabaseName', 'FilterFamily', 'FilterMember')
    _required = ()
    # Descriptor
    DatabaseName = StringEnumDescriptor(
        'DatabaseName', ('BILINEAR', 'CUBIC', 'LAGRANGE', 'NEAREST NEIGHBOR'),
        _required, strict=DEFAULT_STRICT,
        docstring='The filter name.')  # type: str
    FilterFamily = IntegerDescriptor(
        'FilterFamily', _required, strict=DEFAULT_STRICT,
        docstring='The filter family number.')  # type: int
    FilterMember = IntegerDescriptor(
        'FilterMember', _required, strict=DEFAULT_STRICT,
        docstring='The filter member number.')  # type: int

    def __init__(self, DatabaseName=None, FilterFamily=None, FilterMember=None, **kwargs):
        """

        Parameters
        ----------
        DatabaseName : None|str
        FilterFamily : None|int
        FilterMember : None|int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DatabaseName = DatabaseName
        self.FilterFamily = FilterFamily
        self.FilterMember = FilterMember
        super(PredefinedFilterType, self).__init__(**kwargs)


class FilterKernelType(Serializable):
    """
    The filter kernel parameters. Provides the specifics for **either** a predefined or custom
    filter kernel.
    """

    _fields = ('Predefined', 'Custom')
    _required = ()
    _choice = ({'required': True, 'collection': ('Predefined', 'Custom')}, )
    # Descriptor
    Predefined = SerializableDescriptor(
        'Predefined', PredefinedFilterType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PredefinedFilterType
    Custom = StringEnumDescriptor(
        'Custom', ('GENERAL', 'FILTER BANK'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, Predefined=None, Custom=None, **kwargs):
        """

        Parameters
        ----------
        Predefined : None|PredefinedFilterType
        Custom : None|str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Predefined = Predefined
        self.Custom = Custom
        super(FilterKernelType, self).__init__(**kwargs)


class BankCustomType(Serializable, Arrayable):
    """
    A custom filter bank array.
    """
    __slots__ = ('_coefs', )
    _fields = ('Coefs', 'numPhasings', 'numPoints')
    _required = ('Coefs', )
    _numeric_format = {'Coefs': '0.16G'}

    def __init__(self, Coefs=None, **kwargs):
        """

        Parameters
        ----------
        Coefs : numpy.ndarray|list|tuple
        kwargs : dict
        """

        self._coefs = None
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Coefs = Coefs
        super(BankCustomType, self).__init__(**kwargs)

    @property
    def numPhasings(self):
        """
        int: The number of phasings [READ ONLY]
        """

        return self._coefs.shape[0] - 1

    @property
    def numPoints(self):
        """
        int: The number of points [READ ONLY]
        """

        return self._coefs.shape[1] - 1

    @property
    def Coefs(self):
        """
        numpy.ndarray: The two-dimensional filter coefficient array of dtype=float64. Assignment object must be a
        two-dimensional numpy.ndarray, or naively convertible to one.

        .. Note:: this returns the direct coefficient array. Use the `get_array()` method to get a copy of the
            coefficient array of specified data type.
        """

        return self._coefs

    @Coefs.setter
    def Coefs(self, value):
        if value is None:
            raise ValueError('The coefficient array for a BankCustomType instance must be defined.')

        if isinstance(value, (list, tuple)):
            value = numpy.array(value, dtype=numpy.float64)

        if not isinstance(value, numpy.ndarray):
            raise ValueError(
                'Coefs for class BankCustomType must be a list or numpy.ndarray. Received type {}.'.format(type(value)))
        elif len(value.shape) != 2:
            raise ValueError(
                'Coefs for class BankCustomType must be two-dimensional. Received numpy.ndarray '
                'of shape {}.'.format(value.shape))
        elif not value.dtype.name == 'float64':
            value = numpy.cast[numpy.float64](value)
        self._coefs = value

    def __getitem__(self, item):
        return self._coefs[item]

    @classmethod
    def from_array(cls, array):  # type: (numpy.ndarray) -> BankCustomType
        if array is None:
            return None
        return cls(Coefs=array)

    def get_array(self, dtype=numpy.float64):
        """
        Gets **a copy** of the coefficent array of specified data type.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            two-dimensional coefficient array
        """

        return numpy.array(self._coefs, dtype=dtype)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        num_phasings = int_func(node.attrib['numPhasings'])
        num_points = int_func(node.attrib['numPoints'])
        coefs = numpy.zeros((num_phasings+1, num_points+1), dtype=numpy.float64)
        ckey = cls._child_xml_ns_key.get('Coefs', ns_key)
        coef_nodes = find_children(node, 'Coef', xml_ns, ckey)
        for cnode in coef_nodes:
            ind1 = int_func(cnode.attrib['phasing'])
            ind2 = int_func(cnode.attrib['point'])
            val = float(get_node_value(cnode))
            coefs[ind1, ind2] = val
        return cls(Coefs=coefs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        if parent is None:
            parent = doc.getroot()
        if ns_key is None:
            node = create_new_node(doc, tag, parent=parent)
        else:
            node = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)

        if 'Coefs' in self._child_xml_ns_key:
            ctag = '{}:Coef'.format(self._child_xml_ns_key['Coefs'])
        elif ns_key is not None:
            ctag = '{}:Coef'.format(ns_key)
        else:
            ctag = 'Coef'

        node.attrib['numPhasings'] = str(self.numPhasings)
        node.attrib['numPoints'] = str(self.numPoints)
        fmt_func = self._get_formatter('Coefs')
        for i, val1 in enumerate(self._coefs):
            for j, val in enumerate(val1):
                # if val != 0.0:  # should we serialize it sparsely?
                cnode = create_text_node(doc, ctag, fmt_func(val), parent=node)
                cnode.attrib['phasing'] = str(i)
                cnode.attrib['point'] = str(j)
        return node

    def to_dict(self,  check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = OrderedDict()
        out['Coefs'] = self.Coefs.tolist()
        return out


class FilterBankType(Serializable):
    """
    The filter bank type. Provides the specifics for **either** a predefined or custom filter bank.
    """

    _fields = ('Predefined', 'Custom')
    _required = ()
    _choice = ({'required': True, 'collection': ('Predefined', 'Custom')}, )
    # Descriptor
    Predefined = SerializableDescriptor(
        'Predefined', PredefinedFilterType, _required, strict=DEFAULT_STRICT,
        docstring='The predefined filter bank type.')  # type: PredefinedFilterType
    Custom = SerializableDescriptor(
        'Custom', BankCustomType, _required, strict=DEFAULT_STRICT,
        docstring='The custom filter bank.')  # type: BankCustomType

    def __init__(self, Predefined=None, Custom=None, **kwargs):
        """

        Parameters
        ----------
        Predefined : None|PredefinedFilterType
        Custom : None|BankCustomType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Predefined = Predefined
        self.Custom = Custom
        super(FilterBankType, self).__init__(**kwargs)


class FilterType(Serializable):
    """
    Filter parameters for a variety of purposes. Provides **either** the filter bank or
    filter kernel parameters.
    """

    _fields = ('FilterName', 'FilterKernel', 'FilterBank', 'Operation')
    _required = ('FilterName', 'Operation')
    _choice = ({'required': True, 'collection': ('FilterKernel', 'FilterBank')}, )
    # Descriptor
    FilterName = StringDescriptor(
        'FilterName', _required, strict=DEFAULT_STRICT,
        docstring='The name of the filter.')  # type : str
    FilterKernel = SerializableDescriptor(
        'FilterKernel', FilterKernelType, _required, strict=DEFAULT_STRICT,
        docstring='The filter kernel.')  # type: FilterKernelType
    FilterBank = SerializableDescriptor(
        'FilterBank', FilterBankType, _required, strict=DEFAULT_STRICT,
        docstring='The filter bank.')  # type: FilterBankType
    Operation = StringEnumDescriptor(
        'Operation', ('CONVOLUTION', 'CORRELATION'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, FilterName=None, FilterKernel=None, FilterBank=None, Operation=None, **kwargs):
        """

        Parameters
        ----------
        FilterName : str
        FilterKernel : None|FilterKernelType
        FilterBank : None|FilterBankType
        Operation : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.FilterName = FilterName
        self.FilterKernel = FilterKernel
        self.FilterBank = FilterBank
        self.Operation = Operation
        super(FilterType, self).__init__(**kwargs)


################
# NewLookupTableType


class PredefinedLookupType(Serializable):
    """
    The predefined lookup table type. Allows for reference **either** by name, or family/member id number.
    """
    _fields = ('DatabaseName', 'RemapFamily', 'RemapMember')
    _required = ()
    # Descriptor
    DatabaseName = StringDescriptor(
        'DatabaseName', _required, strict=DEFAULT_STRICT,
        docstring='Database name of LUT to use.')  # type: str
    RemapFamily = IntegerDescriptor(
        'RemapFamily', _required, strict=DEFAULT_STRICT,
        docstring='The lookup family number.')  # type: int
    RemapMember = IntegerDescriptor(
        'RemapMember', _required, strict=DEFAULT_STRICT,
        docstring='The lookup member number.')  # type: int

    def __init__(self, DatabaseName=None, RemapFamily=None, RemapMember=None, **kwargs):
        """

        Parameters
        ----------
        DatabaseName : None|str
        RemapFamily : None|int
        RemapMember : None|int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DatabaseName = DatabaseName
        self.RemapFamily = RemapFamily
        self.RemapMember = RemapMember
        super(PredefinedLookupType, self).__init__(**kwargs)


class LUTInfoType(Serializable, Arrayable):
    """
    The lookup table - basically just a one or two dimensional unsigned integer array of bit depth 8 or 16.
    """
    __slots__ = ('_lut_values', )
    _fields = ('LUTValues', 'numLuts', 'size')
    _required = ('LUTValues', 'numLuts', 'size')

    def __init__(self, LUTValues=None, **kwargs):
        """

        Parameters
        ----------
        LUTValues : numpy.ndarray
            The dtype must be `uint8` or `uint16`, and the dimension must be one or two.
        kwargs
        """

        self._lut_values = None
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.LUTValues = LUTValues
        super(LUTInfoType, self).__init__(**kwargs)

    @property
    def LUTValues(self):
        """
        numpy.ndarray: the two dimensional look-up table, where the dtype must be `uint8` or `uint16`.
        The first dimension should correspond to entries (i.e. size of the lookup table), and the
        second dimension should correspond to bands (i.e. number of bands).
        """

        return self._lut_values

    @LUTValues.setter
    def LUTValues(self, value):
        if value is None:
            self._lut_values = None
            return
        if isinstance(value, (tuple, list)):
            value = numpy.array(value, dtype=numpy.uint8)
        if not isinstance(value, numpy.ndarray) or value.dtype.name not in ('uint8', 'uint16'):
            raise ValueError(
                'LUTValues for class LUTInfoType must be a numpy.ndarray of dtype uint8 or uint16.')
        if value.ndim != 2:
            raise ValueError('LUTValues for class LUTInfoType must be two-dimensional.')
        self._lut_values = value

    @property
    def size(self):
        """
        int: the size of each lookup table
        """
        if self._lut_values is None:
            return 0
        else:
            return self._lut_values.shape[0]

    @property
    def numLUTs(self):
        """
        int: The number of lookup tables
        """
        if self._lut_values is None:
            return 0
        else:
            return self._lut_values.shape[1]

    def __len__(self):
        if self._lut_values is None:
            return 0
        return self._lut_values.shape[0]

    def __getitem__(self, item):
        return self._lut_values[item]

    @classmethod
    def from_array(cls, array):
        """
        Create from the lookup table array.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            Must be two-dimensional. If not a numpy.ndarray, this will be naively
            interpreted as `uint8`.

        Returns
        -------
        LUTInfoType
        """
        if array is None:
            return None
        return cls(LUTValues=array)

    def get_array(self, dtype=numpy.uint8):
        """
        Gets **a copy** of the coefficent array of specified data type.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            the lookup table array
        """

        return numpy.array(self._lut_values, dtype=dtype)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        dim1 = int_func(node.attrib['size'])
        dim2 = int_func(node.attrib['numLuts'])
        arr = numpy.zeros((dim1, dim2), dtype=numpy.uint16)

        lut_key = cls._child_xml_ns_key.get('LUTValues', ns_key)
        lut_nodes = find_children(node, 'LUTValues', xml_ns, lut_key)
        for i, lut_node in enumerate(lut_nodes):
            arr[:, i] = [str(el) for el in get_node_value(lut_node)]
        if numpy.max(arr) < 256:
            arr = numpy.cast[numpy.uint8](arr)
        return cls(LUTValues=arr)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        def make_entry(arr):
            value = ' '.join(str(el) for el in arr)
            entry = create_text_node(doc, ltag, value, parent=node)
            entry.attrib['lut'] = str(arr.size)

        if parent is None:
            parent = doc.getroot()

        if ns_key is None:
            node = create_new_node(doc, tag, parent=parent)
        else:
            node = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)

        if 'LUTValues' in self._child_xml_ns_key:
            ltag = '{}:LUTValues'.format(self._child_xml_ns_key['LUTValues'])
        elif ns_key is not None:
            ltag = '{}:LUTValues'.format(ns_key)
        else:
            ltag = 'LUTValues'

        if self._lut_values is not None:
            node.attrib['numLuts'] = str(self.numLUTs)
            node.attrib['size'] = str(self.size)
            if self._lut_values.ndim == 1:
                make_entry(self._lut_values)
            else:
                for j in range(self._lut_values.shape[1]):
                    make_entry(self._lut_values[:, j])
        return node

    def to_dict(self,  check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = OrderedDict()
        if self.LUTValues is not None:
            out['LUTValues'] = self.LUTValues.tolist()
        return out


class CustomLookupType(Serializable):
    """
    A custom lookup table.
    """

    _fields = ('LUTInfo', )
    _required = ('LUTInfo', )
    # Descriptor
    LUTInfo = SerializableDescriptor(
        'LUTInfo', LUTInfoType, _required, strict=DEFAULT_STRICT,
        docstring='The lookup table.')  # type: LUTInfoType

    def __init__(self, LUTInfo=None, **kwargs):
        """

        Parameters
        ----------
        LUTInfo: LUTInfoType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.LUTInfo = LUTInfo
        super(CustomLookupType, self).__init__(**kwargs)


class NewLookupTableType(Serializable):
    """
    The lookup table. Allows **either** a reference to a prefined lookup table, or
    custom lookup table array.
    """

    _fields = ('LUTName', 'Predefined', 'Custom')
    _required = ('LUTName', )
    _choice = ({'required': True, 'collection': ('Predefined', 'Custom')}, )
    # Descriptor
    LUTName = StringDescriptor(
        'LUTName', _required, strict=DEFAULT_STRICT,
        docstring='The lookup table name')  # type: str
    Predefined = SerializableDescriptor(
        'Predefined', PredefinedLookupType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PredefinedLookupType
    Custom = SerializableDescriptor(
        'Custom', CustomLookupType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: CustomLookupType

    def __init__(self, LUTName=None, Predefined=None, Custom=None, **kwargs):
        """

        Parameters
        ----------
        LUTName : str
        Predefined : None|PredefinedLookupType
        Custom : None|CustomLookupType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.LUTName = LUTName
        self.Predefined = Predefined
        self.Custom = Custom
        super(NewLookupTableType, self).__init__(**kwargs)
