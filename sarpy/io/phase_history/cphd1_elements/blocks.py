"""
Basic building blocks for CPHD standard - mostly overlap with SICD elements
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, List

import numpy

from sarpy.io.xml.base import Serializable, Arrayable, SerializableArray
from sarpy.io.xml.descriptors import SerializableDescriptor, SerializableArrayDescriptor, \
    IntegerDescriptor, FloatDescriptor

from .base import DEFAULT_STRICT


###################
# module variables
POLARIZATION_TYPE = ('X', 'Y', 'V', 'H', 'RHC', 'LHC', 'UNSPECIFIED')


class LSType(Serializable, Arrayable):
    """
    Represents line and sample.
    """

    _fields = ('Line', 'Sample')
    _required = _fields
    _numeric_format = {'Line': '0.16G', 'Sample': '0.16G'}
    # Descriptor
    Line = FloatDescriptor(
        'Line', _required, strict=DEFAULT_STRICT,
        docstring='The Line.')  # type: float
    Sample = FloatDescriptor(
        'Sample', _required, strict=DEFAULT_STRICT,
        docstring='The Sample.')  # type: float

    def __init__(self, Line=None, Sample=None, **kwargs):
        """

        Parameters
        ----------
        Line : float
        Sample : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Line = Line
        self.Sample = Sample
        super(LSType, self).__init__(**kwargs)

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
            array of the form [Line, Sample]
        """

        return numpy.array([self.Line, self.Sample], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Construct from a iterable.

        Parameters
        ----------
        array : numpy.ndarray|list|tuple

        Returns
        -------
        LSType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Line=array[0], Sample=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LSVertexType(LSType):
    """
    An array element of LSType.
    """

    _fields = ('Line', 'Sample', 'index')
    _required = _fields
    # descriptors
    index = IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='The array index.')  # type: int

    def __init__(self, Line=None, Sample=None, index=None, **kwargs):
        """

        Parameters
        ----------
        Line : float
        Sample : float
        index : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(LSVertexType, self).__init__(Line=Line, Sample=Sample, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Line, Sample]
        index : int
            array index
        Returns
        -------
        XYVertexType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Line=array[0], Sample=array[1], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class XYType(Serializable, Arrayable):
    """
    A point in two-dimensional spatial coordinates.
    """

    _fields = ('X', 'Y')
    _required = _fields
    _numeric_format = {'X': '0.16G', 'Y': '0.16G'}
    # descriptors
    X = FloatDescriptor(
        'X', _required, strict=True,
        docstring='The X attribute. Assumed to ECF or other, similar '
                  'coordinates.')  # type: float
    Y = FloatDescriptor(
        'Y', _required, strict=True,
        docstring='The Y attribute. Assumed to ECF or other, similar '
                  'coordinates.')  # type: float

    def __init__(self, X=None, Y=None, **kwargs):
        """
        Parameters
        ----------
        X : float
        Y : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.X, self.Y = X, Y
        super(XYType, self).__init__(**kwargs)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [X, Y]

        Returns
        -------
        XYType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(X=array[0], Y=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))

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
            array of the form [X,Y]
        """

        return numpy.array([self.X, self.Y], dtype=dtype)


class XYVertexType(XYType):
    """
    An array element of XYType.
    """

    _fields = ('X', 'Y', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='The array index.')  # type: int

    def __init__(self, X=None, Y=None, index=None, **kwargs):
        """

        Parameters
        ----------
        X : float
        Y : float
        index : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(XYVertexType, self).__init__(X=X, Y=Y, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [X, Y]
        index : int
            array index
        Returns
        -------
        XYVertexType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(X=array[0], Y=array[1], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class AreaType(Serializable):
    """
    An area object.
    """

    _fields = ('X1Y1', 'X2Y2', 'Polygon')
    _required = _fields
    _collections_tags = {'Polygon': {'array': True, 'child_tag': 'Vertex'}}
    # descriptors
    X1Y1 = SerializableDescriptor(
        'X1Y1', XYType, _required, strict=DEFAULT_STRICT,
        docstring='*"Minimum"* corner of the rectangle in Image '
                  'coordinates.')  # type: XYType
    X2Y2 = SerializableDescriptor(
        'X2Y2', XYType, _required, strict=DEFAULT_STRICT,
        docstring='*"Maximum"* corner of the rectangle in Image '
                  'coordinates.')  # type: XYType
    Polygon = SerializableArrayDescriptor(
        'Polygon', XYVertexType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='Polygon further reducing the bounding box, in Image '
                  'coordinates.')  # type: Union[SerializableArray, List[XYVertexType]]

    def __init__(self, X1Y1=None, X2Y2=None, Polygon=None, **kwargs):
        """

        Parameters
        ----------
        X1Y1 : XYType|numpy.ndarray|list|tuple
        X2Y2 : XYType|numpy.ndarray|list|tuple
        Polygon : SerializableArray|List[XYVertexType]|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.X1Y1 = X1Y1
        self.X2Y2 = X2Y2
        self.Polygon = Polygon
        super(AreaType, self).__init__(**kwargs)
