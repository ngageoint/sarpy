# -*- coding: utf-8 -*-
"""
Multipurpose SIDD elements
"""

from collections import OrderedDict

import numpy

from .base import Serializable, Arrayable, _SerializableDescriptor, _SerializableListDescriptor, \
    _IntegerDescriptor, _IntegerListDescriptor, _StringDescriptor, _StringEnumDescriptor, \
    _ParametersDescriptor, DEFAULT_STRICT, ParametersCollection, int_func, \
    _get_node_value, _create_text_node, _create_new_node

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


# class _(Serializable):
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
#         super(_, self).__init__(**kwargs)


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

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self._coefs = None
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
        self.Predefined = Predefined
        self.Custom = Custom
        super(FilterBankType, self).__init__(**kwargs)


class FilterType(Serializable):
    """

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
        self.FilterName = FilterName
        self.FilterKernel = FilterKernel
        self.FilterBank = FilterBank
        self.Operation = Operation
        super(FilterType, self).__init__(**kwargs)


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
        self.DatabaseName = DatabaseName
        self.RemapFamily = RemapFamily
        self.RemapMember = RemapMember
        super(PredefinedLookupType, self).__init__(**kwargs)


class LUTInfoType(Serializable):
    """

    """
    _fields = ( )
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(LUTInfoType, self).__init__(**kwargs)


class CustomLookupType(Serializable):
    """

    """
    _fields = ( )
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(CustomLookupType, self).__init__(**kwargs)


class NewLookupTableType(Serializable):
    """

    """
    _fields = ()
    _required = ()
    # Descriptor

    # TODO:

    def __init__(self, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        super(NewLookupTableType, self).__init__(**kwargs)
