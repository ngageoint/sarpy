# -*- coding: utf-8 -*-
"""
Multipurpose SIDD elements
"""

from collections import OrderedDict
from typing import Union

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, Arrayable, _SerializableDescriptor, \
    _IntegerDescriptor, _FloatDescriptor, _FloatModularDescriptor, _StringDescriptor, _StringEnumDescriptor, \
    int_func, _get_node_value, _create_text_node, _create_new_node
from ..sicd_elements.blocks import XYZType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RangeAzimuthType(Serializable, Arrayable):
    """
    Represents range and azimuth.
    """
    _fields = ('Range', 'Azimuth')
    _required = ('Range', 'Azimuth')
    # Descriptor
    Range = _FloatDescriptor(
        'Range', _required, strict=DEFAULT_STRICT,
        docstring='The range in meters.')  # type: float
    Azimuth = _FloatDescriptor(
        'Azimuth', _required, strict=DEFAULT_STRICT,
        docstring='The azimuth in degrees.')  # type: float

    def __init__(self, Range=None, Azimuth=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.Range = Range
        self.Azimuth = Azimuth
        super(RangeAzimuthType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : numpy.dtype
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
    # Descriptor
    Angle = _FloatModularDescriptor(
        'Angle', 180.0, _required, strict=DEFAULT_STRICT,
        docstring='The angle.')  # type: float
    Magnitude = _FloatDescriptor(
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
        self.Angle = Angle
        self.Magnitude = Magnitude
        super(AngleMagnitudeType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : numpy.dtype
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

        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Angle=array[0], Magnitude=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


##############
# Reference Point

class RowColDoubleType(Serializable, Arrayable):
    _fields = ('Row', 'Col')
    _required = _fields
    Row = _FloatDescriptor(
        'Row', _required, strict=True, docstring='The Row attribute.')  # type: float
    Col = _FloatDescriptor(
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
        self.Row, self.Col = Row, Col
        super(RowColDoubleType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : numpy.dtype
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
        RowColType
        """

        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Row=array[0], Col=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class ReferencePointType(Serializable):
    """
    A reference point.
    """

    _fields = ('ECEF', 'Point', 'name')
    _required = ('ECEF', 'Point')
    _set_as_attribute = ('name', )
    # Descriptor
    ECEF = _SerializableDescriptor(
        'ECEF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The EXEF coordinates of the reference point.')  # type: XYZType
    Point = _SerializableDescriptor(
        'Point', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='The pixel coordinates of the reference point.')  # type: RowColDoubleType
    name = _StringDescriptor(
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
        self.ECEF = ECEF
        self.Point = Point
        self.name = name
        super(ReferencePointType, self).__init__(**kwargs)


#################
# Filter Type

class PredefinedFilterType(Serializable):
    """
    The predefined filter type.
    """
    _fields = ('DatabaseName', 'FilterFamily', 'FilterMember')
    _required = ()
    # Descriptor
    DatabaseName = _StringEnumDescriptor(
        'DatabaseName', ('BILINEAR', 'CUBIC', 'LAGRANGE', 'NEAREST NEIGHBOR'),
        _required, strict=DEFAULT_STRICT,
        docstring='The filter name.')  # type: str
    FilterFamily = _IntegerDescriptor(
        'FilterFamily', _required, strict=DEFAULT_STRICT,
        docstring='The filter family number.')  # type: int
    FilterMember = _IntegerDescriptor(
        'FilterMember', _required, strict=DEFAULT_STRICT,
        docstring='The filter member number.')  # type: int

    def __init__(self, DatabaseName=None, FilterFamily=None, FilterMember=None, **kwargs):
        """

        Parameters
        ----------
        DatabaseName : str
        FilterFamily : int
        FilterMember : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.DatabaseName = DatabaseName
        self.FilterFamily = FilterFamily
        self.FilterMember = FilterMember
        super(PredefinedFilterType, self).__init__(**kwargs)


class FilterKernelType(Serializable):
    """
    The filter kernel parameters.
    """

    _fields = ('Predefined', 'Custom')
    _required = ()
    _choice = ({'required': True, 'collection': ('Predefined', 'Custom')}, )
    # Descriptor
    Predefined = _SerializableDescriptor(
        'Predefined', PredefinedFilterType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PredefinedFilterType
    Custom = _StringEnumDescriptor(
        'Custom', ('GENERAL', 'FILTER BANK'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, Predefined=None, Custom=None, **kwargs):
        """

        Parameters
        ----------
        Predefined : PredefinedFilterType
        Custom : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
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
        return cls(Coefs=array)

    def get_array(self, dtype=numpy.float64):
        """
        Gets **a copy** of the coefficent array of specified data type.

        Parameters
        ----------
        dtype : numpy.dtype
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            two-dimensional coefficient array
        """

        return numpy.array(self._coefs, dtype=dtype)

    @classmethod
    def from_node(cls, node, xml_ns, kwargs=None):
        numPhasings = int_func(node.attrib['numPhasings'])
        numPoints = int_func(node.attrib['numPoints'])
        coefs = numpy.zeros((numPhasings+1, numPoints+1), dtype=numpy.float64)
        coef_nodes = node.findall('Coef') if xml_ns is None else node.findall('default:Coef', xml_ns)
        for cnode in coef_nodes:
            ind1 = int_func(cnode.attrib['phasing'])
            ind2 = int_func(cnode.attrib['point'])
            val = float(_get_node_value(cnode))
            coefs[ind1, ind2] = val
        return cls(Coefs=coefs)

    def to_node(self, doc, tag, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        if parent is None:
            parent = doc.getroot()
        node = _create_new_node(doc, tag, parent=parent)
        node.attrib['numPhasings'] = str(self.numPhasings)
        node.attrib['numPoints'] = str(self.numPoints)
        fmt_func = self._get_formatter('Coefs')
        for i, val1 in enumerate(self._coefs):
            for j, val in enumerate(val1):
                # if val != 0.0:  # should we serialize it sparsely?
                cnode = _create_text_node(doc, 'Coef', fmt_func(val), parent=node)
                cnode.attrib['phasing'] = str(i)
                cnode.attrib['point'] = str(j)
        return node

    def to_dict(self,  check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = OrderedDict()
        out['Coefs'] = self.Coefs.tolist()
        return out


class FilterBankType(Serializable):
    """
    The filter bank type.
    """

    _fields = ('Predefined', 'Custom')
    _required = ()
    _choice = ({'required': True, 'collection': ('Predefined', 'Custom')}, )
    # Descriptor
    Predefined = _SerializableDescriptor(
        'Predefined', PredefinedFilterType, _required, strict=DEFAULT_STRICT,
        docstring='The predefined filter bank type.')  # type: PredefinedFilterType
    Custom = _SerializableDescriptor(
        'Custom', BankCustomType, _required, strict=DEFAULT_STRICT,
        docstring='The custom filter bank.')  # type: BankCustomType

    def __init__(self, Predefined=None, Custom=None, **kwargs):
        """

        Parameters
        ----------
        Predefined : PredefinedFilterType
        Custom : BankCustomType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(FilterBankType, self).__init__(Predefined=Predefined, Custom=Custom, **kwargs)


class FilterType(Serializable):
    """
    Filter parameters for a variety of purposes.
    """

    _fields = ('FilterName', 'FilterKernel', 'FilterBank', 'Operation')
    _required = ('FilterName', 'Operation')
    _choice = ({'required': True, 'collection': ('FilterKernel', 'FilterBank')}, )
    # Descriptor
    FilterName = _StringDescriptor(
        'FilterName', _required, strict=DEFAULT_STRICT,
        docstring='The name of the filter.')  # type : str
    FilterKernel = _SerializableDescriptor(
        'FilterKernel', FilterKernelType, _required, strict=DEFAULT_STRICT,
        docstring='The filter kernel.')  # type: FilterKernelType
    FilterBank = _SerializableDescriptor(
        'FilterBank', FilterBankType, _required, strict=DEFAULT_STRICT,
        docstring='The filter bank.')  # type: FilterBankType
    Operation = _StringEnumDescriptor(
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
        super(FilterType, self).__init__(
            FilterName=FilterName, FilterKernel=FilterKernel, FilterBank=FilterBank, Operation=Operation, **kwargs)


################
# NewLookupTableType


class PredefinedLookupType(Serializable):
    """
    The predefined lookup table type.
    """
    _fields = ('DatabaseName', 'RemapFamily', 'RemapMember')
    _required = ()
    # Descriptor
    DatabaseName = _StringDescriptor(
        'DatabaseName', _required, strict=DEFAULT_STRICT,
        docstring='Database name of LUT to use.')  # type: str
    RemapFamily = _IntegerDescriptor(
        'RemapFamily', _required, strict=DEFAULT_STRICT,
        docstring='The lookup family number.')  # type: int
    RemapMember = _IntegerDescriptor(
        'RemapMember', _required, strict=DEFAULT_STRICT,
        docstring='The lookup member number.')  # type: int

    def __init__(self, DatabaseName=None, RemapFamily=None, RemapMember=None, **kwargs):
        """

        Parameters
        ----------
        DatabaseName : str
        RemapFamily : int
        RemapMember : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(PredefinedLookupType, self).__init__(
            DatabaseName=DatabaseName, RemapFamily=RemapFamily, RemapMember=RemapMember, **kwargs)


class LUTInfoType(Serializable, Arrayable):
    """
    The lookup table - basically just a one or two dimensional unsigned integer array of bit depth 8 or 16.
    """
    __slots__ = ('_lut_values', )
    _fields = ('LUTValues', 'numLuts', 'size')
    _required = ('LUTValues', )

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

        return cls(LUTValues=array)

    def get_array(self, dtype=numpy.uint8):
        """
        Gets **a copy** of the coefficent array of specified data type.

        Parameters
        ----------
        dtype : numpy.dtype
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            the lookup table array
        """

        return numpy.array(self._lut_values, dtype=dtype)

    @classmethod
    def from_node(cls, node, xml_ns, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        xml_ns : dict
            The xml namespace dictionary.
        kwargs : None|dict
            `None` or dictionary of previously serialized attributes. For use in inheritance call, when certain
            attributes require specific deserialization.

        Returns
        -------
        LUTInfoType
            corresponding class instance
        """

        dim1 = int_func(node.attrib['size'])
        dim2 = int_func(node.attrib['numLuts'])
        arr = numpy.zeros((dim1, dim2), dtype=numpy.uint16)
        lut_nodes = node.findall('LUTValues') if xml_ns is None else node.findall('default:LUTValues', xml_ns)
        for i, lut_node in enumerate(lut_nodes):
            arr[:, i] = [str(el) for el in _get_node_value(lut_node)]
        if numpy.max(arr) < 256:
            arr = numpy.cast[numpy.uint8](arr)
        return cls(LUTValues=arr)

    def to_node(self, doc, tag, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        def make_entry(arr):
            value = ' '.join(str(el) for el in arr)
            entry = _create_text_node(doc, 'LUTValues', value, parent=node)
            entry.attrib['lut'] = str(arr.size)

        if self._lut_values is None or self._lut_values.ndim == 0:
            return

        if parent is None:
            parent = doc.getroot()
        node = _create_new_node(doc, tag, parent=parent)
        node.attrib['numLuts'] = str(self.numLUTs)
        node.attrib['size'] = str(self.size)
        if self._lut_values.ndim == 1:
            make_entry(self._lut_values)
        else:
            for j in range(self._lut_values.shape[1]):
                make_entry(self._lut_values[:, j])

    def to_dict(self,  check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = OrderedDict()
        out['LUTValues'] = self.LUTValues.tolist()
        return out


class CustomLookupType(Serializable):
    """
    A custom lookup table.
    """

    _fields = ('LUTInfo', )
    _required = ('LUTInfo', )
    # Descriptor
    LUTInfo = _SerializableDescriptor(
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
        self.LUTInfo = LUTInfo
        super(CustomLookupType, self).__init__(**kwargs)


class NewLookupTableType(Serializable):
    """

    """
    _fields = ('Predefined', 'Custom')
    _required = ()
    # Descriptor
    Predefined = _SerializableDescriptor(
        'Predefined', PredefinedLookupType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PredefinedLookupType
    Custom = _SerializableDescriptor(
        'Custom', CustomLookupType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: CustomLookupType

    def __init__(self, Predefined=None, Custom=None, **kwargs):
        """

        Parameters
        ----------
        Predefined : PredefinedLookupType
        Custom : CustomLookupType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.Predefined = Predefined
        self.Custom = Custom
        super(NewLookupTableType, self).__init__(**kwargs)


############

# class dummy(Serializable):
#     """
#
#     """
#     _fields = ()
#     _required = ()
#     # Descriptor
#
#     # TODO:
#
#     def __init__(self, **kwargs):
#         if '_xml_ns' in kwargs:
#             self._xml_ns = kwargs['_xml_ns']
#         super(dummy, self).__init__(**kwargs)
