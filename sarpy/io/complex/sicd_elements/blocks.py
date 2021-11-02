"""
Basic building blocks for SICD standard.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from collections import OrderedDict

import numpy

from sarpy.io.xml.base import Serializable, Arrayable, \
    get_node_value, create_text_node, create_new_node, find_children
from sarpy.io.xml.descriptors import IntegerDescriptor, StringEnumDescriptor, \
    FloatDescriptor, FloatModularDescriptor, SerializableDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT


#########
# Polarization constants
POLARIZATION1_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER', 'UNKNOWN', 'SEQUENCE')
POLARIZATION2_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER')
DUAL_POLARIZATION_VALUES = (
    'V:V', 'V:H', 'V:RHC', 'V:LHC',
    'H:V', 'H:H', 'H:RHC', 'H:LHC',
    'RHC:V', 'RHC:H', 'RHC:RHC', 'RHC:LHC',
    'LHC:V', 'LHC:H', 'LHC:RHC', 'LHC:LHC',
    'OTHER', 'UNKNOWN')


##########
# Geographical coordinates


class XYZType(Serializable, Arrayable):
    """A spatial point in ECF coordinates."""
    _fields = ('X', 'Y', 'Z')
    _required = _fields
    _numeric_format = {'X': FLOAT_FORMAT, 'Y': FLOAT_FORMAT, 'Z': FLOAT_FORMAT}
    # descriptors
    X = FloatDescriptor(
        'X', _required, strict=True,
        docstring='The X attribute. Assumed to ECF or other, similar coordinates.')  # type: float
    Y = FloatDescriptor(
        'Y', _required, strict=True,
        docstring='The Y attribute. Assumed to ECF or other, similar coordinates.')  # type: float
    Z = FloatDescriptor(
        'Z', _required, strict=True,
        docstring='The Z attribute. Assumed to ECF or other, similar coordinates.')  # type: float

    def __init__(self, X=None, Y=None, Z=None, **kwargs):
        """
        Parameters
        ----------
        X : float
        Y : float
        Z : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.X, self.Y, self.Z = X, Y, Z
        super(XYZType, self).__init__(**kwargs)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [X, Y, Z]

        Returns
        -------
        XYZType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(X=array[0], Y=array[1], Z=array[2])
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
            array of the form [X,Y,Z]
        """

        return numpy.array([self.X, self.Y, self.Z], dtype=dtype)


class LatLonType(Serializable, Arrayable):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon')
    _required = _fields
    _numeric_format = {'Lat': FLOAT_FORMAT, 'Lon': FLOAT_FORMAT}
    # descriptors
    Lat = FloatDescriptor(
        'Lat', _required, strict=True,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')  # type: float
    Lon = FloatDescriptor(
        'Lon', _required, strict=True,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')  # type: float

    def __init__(self, Lat=None, Lon=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Lat, self.Lon = Lat, Lon
        super(LatLonType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64, order='LAT'):
        """
        Gets an array representation of the data.

        Parameters
        ----------
        order : str
            Determines array order. 'LAT' yields [Lat, Lon], and anything else yields [Lon, Lat].
        dtype : str|numpy.dtype|numpy.number
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon]

        Returns
        -------
        LatLonType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))

    def dms_format(self, frac_secs=False):
        """
        Get degree-minutes-seconds representation.
        Parameters
        ----------
        frac_secs : bool
            Should a fractional seconds (i.e. a float), otherwise integer

        Returns
        -------
        tuple
            of the form ((deg lat, mins lat, secs lat, N/S), (deg lon, mins lon, secs lon, E/W))
        Here degrees and minutes will be int, secs will be float.
        """

        def reduce(value):
            val = abs(value)
            deg = int(val)
            val = 60*(val - deg)
            mins = int(val)
            secs = 60*(val - mins)
            if not frac_secs:
                secs = int(secs)
            return deg, mins, secs

        x = 'S' if self.Lat < 0 else 'N'
        y = 'W' if self.Lon < 0 else 'E'
        return reduce(self.Lat) + (x, ), reduce(self.Lon) + (y, )


class LatLonArrayElementType(LatLonType):
    """An geographic point in an array"""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    index = IntegerDescriptor(
        'index', _required, strict=False, docstring="The array index")  # type: int

    def __init__(self, Lat=None, Lon=None, index=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(LatLonArrayElementType, self).__init__(Lat=Lat, Lon=Lon, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon]
        index : int
            array index
        Returns
        -------
        LatLonArrayElementType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonRestrictionType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon')
    _required = _fields
    # descriptors
    Lat = FloatModularDescriptor(
        'Lat', 90.0, _required, strict=True,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')  # type: float
    Lon = FloatModularDescriptor(
        'Lon', 180.0, _required, strict=True,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')  # type: float

    def __init__(self, Lat=None, Lon=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(LatLonRestrictionType, self).__init__(Lat=Lat, Lon=Lon, **kwargs)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon]

        Returns
        -------
        LatLonRestrictionType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonHAEType(LatLonType):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon', 'HAE')
    _required = _fields
    _numeric_format = {'Lat': FLOAT_FORMAT, 'Lon': FLOAT_FORMAT, 'HAE': FLOAT_FORMAT}
    # descriptors
    HAE = FloatDescriptor(
        'HAE', _required, strict=True,
        docstring='The Height Above Ellipsoid (in meters) attribute. Assumed to be '
                  'WGS-84 coordinates.')  # type: float

    def __init__(self, Lat=None, Lon=None, HAE=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        HAE : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.HAE = HAE
        super(LatLonHAEType, self).__init__(Lat=Lat, Lon=Lon, **kwargs)

    def get_array(self, dtype=numpy.float64, order='LAT'):
        """
        Gets an array representation of the data.

        Parameters
        ----------
        order : str
            Determines array order. 'LAT' yields [Lat, Lon, HAE], and anything else yields  [Lon, Lat, HAE].
        dtype : str|numpy.dtype|numpy.number
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon, self.HAE], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat, self.HAE], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon, HAE]

        Returns
        -------
        LatLonHAEType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], HAE=array[2])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonHAERestrictionType(LatLonHAEType):
    _fields = ('Lat', 'Lon', 'HAE')
    _required = _fields
    """A three-dimensional geographic point in WGS-84 coordinates."""
    Lat = FloatModularDescriptor(
        'Lat', 90.0, _required, strict=True,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')  # type: float
    Lon = FloatModularDescriptor(
        'Lon', 180.0, _required, strict=True,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')  # type: float

    def __init__(self, Lat=None, Lon=None, HAE=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        HAE : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(LatLonHAERestrictionType, self).__init__(Lat=Lat, Lon=Lon, HAE=HAE, **kwargs)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon, HAE]

        Returns
        -------
        LatLonHAERestrictionType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], HAE=array[2])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonCornerType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = IntegerDescriptor(
        'index', _required, strict=False, bounds=(1, 4),
        docstring='The integer index. This represents a clockwise enumeration of '
                  'the rectangle vertices wrt the frame of reference of the collector. '
                  'Should be 1-4, but 0-3 may be permissible.')  # type: int

    def __init__(self, Lat=None, Lon=None, index=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(LatLonCornerType, self).__init__(Lat=Lat, Lon=Lon, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon]
        index : int
            array index
        Returns
        -------
        LatLonCornerType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected coords to be of length 2, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonCornerStringType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # other specific class variable
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = StringEnumDescriptor(
        'index', _CORNER_VALUES, _required, strict=False,
        docstring="The string index.")  # type: str

    def __init__(self, Lat=None, Lon=None, index=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        index : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(LatLonCornerStringType, self).__init__(Lat=Lat, Lon=Lon, **kwargs)

    @classmethod
    def from_array(cls, array, index='1:FRFC'):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon]
        index : str
            array index in  ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
        Returns
        -------
        LatLonCornerStringType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonHAECornerRestrictionType(LatLonHAERestrictionType):
    """A three-dimensional geographic point in WGS-84 coordinates. Represents a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'HAE', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = IntegerDescriptor(
        'index', _required, strict=False, bounds=(1, 4),
        docstring='The integer index. This represents a clockwise enumeration of the '
                  'rectangle vertices wrt the frame of reference of the collector. '
                  'Should be 1-4, but 0-3 may be permissible.')  # type: int

    def __init__(self, Lat=None, Lon=None, HAE=None, index=None, **kwargs):
        """

        Parameters
        ----------
        Lat : float
        Lon : float
        HAE : float
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(LatLonHAECornerRestrictionType, self).__init__(Lat=Lat, Lon=Lon, HAE=HAE, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon, HAE]
        index : int
            array index
        Returns
        -------
        LatLonHAECornerRestrictionType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], HAE=array[2], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class LatLonHAECornerStringType(LatLonHAEType):
    """A three-dimensional geographic point in WGS-84 coordinates. Represents a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'HAE', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = StringEnumDescriptor(
        'index', _CORNER_VALUES, _required, strict=False, docstring="The string index.")  # type: str

    def __init__(self, Lat=None, Lon=None, HAE=None, index=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        HAE : float
        index : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(LatLonHAECornerStringType, self).__init__(Lat=Lat, Lon=Lon, HAE=HAE, **kwargs)

    @classmethod
    def from_array(cls, array, index='1:FRFC'):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon, HAE]
        index : str
            array index in ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
        Returns
        -------
        LatLonHAECornerStringType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], HAE=array[2], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


#######
# Image space coordinates


class RowColType(Serializable, Arrayable):
    """A row and column attribute container - used as indices into array(s)."""
    _fields = ('Row', 'Col')
    _required = _fields
    Row = IntegerDescriptor(
        'Row', _required, strict=True, docstring='The Row attribute.')  # type: int
    Col = IntegerDescriptor(
        'Col', _required, strict=True, docstring='The Column attribute.')  # type: int

    def __init__(self, Row=None, Col=None, **kwargs):
        """
        Parameters
        ----------
        Row : int
        Col : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Row, self.Col = Row, Col
        super(RowColType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.int64):
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
        RowColType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Row=array[0], Col=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class RowColArrayElement(RowColType):
    """A array element row and column attribute container - used as indices into other array(s)."""
    # Note - in the SICD standard this type is listed as RowColvertexType. This is not a descriptive name
    # and has an inconsistency in camel case
    _fields = ('Row', 'Col', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = IntegerDescriptor(
        'index', _required, strict=False, docstring='The array index attribute.')  # type: int

    def __init__(self, Row=None, Col=None, index=None, **kwargs):
        """
        Parameters
        ----------
        Row : int
        Col : int
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(RowColArrayElement, self).__init__(Row=Row, Col=Col, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Row, Col]
        index : int
            the array index

        Returns
        -------
        RowColArrayElement
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Row=array[0], Col=array[1], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


###############
# Polynomial Types


class Poly1DType(Serializable, Arrayable):
    """
    Represents a one-variable polynomial, defined by one-dimensional coefficient array.
    """

    __slots__ = ('_coefs', )
    _fields = ('Coefs', 'order1')
    _required = ('Coefs', )
    _numeric_format = {'Coefs': FLOAT_FORMAT}

    def __init__(self, Coefs=None, **kwargs):
        """

        Parameters
        ----------
        Coefs : numpy.ndarray|tuple|list
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self._coefs = None
        self.Coefs = Coefs
        super(Poly1DType, self).__init__(**kwargs)

    @property
    def order1(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent presented in the monomial terms of coefs.
        """

        return self.Coefs.size - 1

    @property
    def Coefs(self):
        """
        numpy.ndarray: The one-dimensional polynomial coefficient array of dtype=float64. Assignment object must be a
        one-dimensional numpy.ndarray, or naively convertible to one.

        .. Note:: this returns the direct coefficient array. Use the `get_array()` method to get a copy of the
            coefficient array of specified data type.
        """

        return self._coefs

    @Coefs.setter
    def Coefs(self, value):
        if value is None:
            raise ValueError('The coefficient array for a Poly1DType instance must be defined.')

        if isinstance(value, (list, tuple)):
            value = numpy.array(value, dtype=numpy.float64)

        if not isinstance(value, numpy.ndarray):
            raise ValueError(
                'Coefs for class Poly1D must be a list or numpy.ndarray. Received type {}.'.format(type(value)))
        elif len(value.shape) != 1:
            raise ValueError(
                'Coefs for class Poly1D must be one-dimensional. Received numpy.ndarray '
                'of shape {}.'.format(value.shape))
        elif not value.dtype.name == 'float64':
            value = numpy.cast[numpy.float64](value)
        self._coefs = value

    def __call__(self, x):
        """
        Evaluate the polynomial at points `x`. This passes `x` straight through to :func:`polyval` of
        `numpy.polynomial.polynomial`.

        Parameters
        ----------
        x : float|int|numpy.ndarray
            The point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        return numpy.polynomial.polynomial.polyval(x, self._coefs)

    def __getitem__(self, item):
        return self._coefs[item]

    def __setitem__(self, item, value):
        self._coefs[item] = value

    def derivative(self, der_order=1, return_poly=False):
        """
        Calculate the `der_order` derivative of the polynomial.

        Parameters
        ----------
        der_order : int
            the order of the derivative
        return_poly : bool
            return a Poly1DType if True, otherwise return the coefficient array.

        Returns
        -------
        Poly1DType|numpy.ndarray
        """

        coefs = numpy.polynomial.polynomial.polyder(self._coefs, der_order)
        if return_poly:
            return Poly1DType(Coefs=coefs)
        return coefs

    def derivative_eval(self, x, der_order=1):
        """
        Evaluate the `der_order` derivative of the polynomial at points `x`. This uses the
        functionality presented in `numpy.polynomial.polynomial`.

        Parameters
        ----------
        x : float|int|numpy.ndarray
            The point(s) at which to evaluate.
        der_order : int
            The derivative.
        Returns
        -------
        numpy.ndarray
        """

        coefs = self.derivative(der_order=der_order, return_poly=False)
        return numpy.polynomial.polynomial.polyval(x, coefs)

    def shift(self, t_0, alpha=1, return_poly=False):
        r"""
        Transform a polynomial with respect to a affine shift in the coordinate system.
        That is, :math:`P(x) = Q(\alpha\cdot(t-t_0))`.

        Be careful to follow the convention that the transformation parameters express the *current coordinate system*
        as a shifted, **and then** scaled version of the *new coordinate system*. If the new coordinate is
        :math:`t = \beta\cdot x - t_0`, then :math:`x = (t - t_0)/\beta`, and :math:`\alpha = 1/\beta`.

        Parameters
        ----------
        t_0 : float
            the **current center coordinate** in the **new coordinate system.**
            That is, `x=0` when `t=t_0`.

        alpha : float
            the scale. That is, when `t = t0 + 1`, then `x = alpha`. **NOTE:** it is assumed that the
            coordinate system is re-centered, and **then** scaled.

        return_poly : bool
            if `True`, a Poly1DType object be returned, otherwise the coefficients array is returned.

        Returns
        -------
        Poly1DType|numpy.ndarray
        """
        # prepare array workspace
        out = numpy.copy(self._coefs)
        if t_0 != 0 and out.size > 1:
            siz = out.size
            # let's use horner's method, so iterate from top down
            for i in range(siz):
                index = siz-i-1
                if i > 0:
                    out[index:siz-1] -= t_0*out[index+1:siz]

        if alpha != 1 and out.size > 1:
            out *= numpy.power(alpha, numpy.arange(out.size))

        if return_poly:
            return Poly1DType(Coefs=out)
        else:
            return out

    @classmethod
    def from_array(cls, array):
        """
        Create from the coefficients array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            must be one-dimensional

        Returns
        -------
        Poly1DType
        """
        if array is None:
            return None
        return cls(Coefs=array)

    def get_array(self, dtype=numpy.float64):
        """
        Gets *a copy* of the coefficent array of specified data type.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            one-dimensional coefficient array
        """

        return numpy.array(self._coefs, dtype=dtype)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        order1 = int(node.attrib['order1'])
        coefs = numpy.zeros((order1+1, ), dtype=numpy.float64)

        coef_key = cls._child_xml_ns_key.get('Coefs', ns_key)
        coef_nodes = find_children(node, 'Coef', xml_ns, coef_key)
        for cnode in coef_nodes:
            ind = int(cnode.attrib['exponent1'])
            val = float(get_node_value(cnode))
            coefs[ind] = val
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

        node.attrib['order1'] = str(self.order1)
        fmt_func = self._get_formatter('Coef')
        for i, val in enumerate(self.Coefs):
            # if val != 0.0:  # should we serialize it sparsely?
            cnode = create_text_node(doc, ctag, fmt_func(val), parent=node)
            cnode.attrib['exponent1'] = str(i)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = OrderedDict()
        out['Coefs'] = self.Coefs.tolist()
        return out

    def minimize_order(self):
        """
        Trim the trailing zeros for the coefficient array. This modifies the object in place.

        Returns
        -------
        None
        """

        boolc = (self.Coefs != 0)
        if not numpy.any(boolc):
            self.Coefs = numpy.zeros((1, ), dtype='float64')
            return
        last_ind = numpy.amax(numpy.arange(self.Coefs.size)[boolc])
        if last_ind == self.Coefs.size-1:
            return

        if last_ind == 0:
            self.Coefs = numpy.array([self.Coefs[0], ], dtype='float64')
        else:
            self.Coefs = self.Coefs[:last_ind+1]


class Poly2DType(Serializable, Arrayable):
    """
    Represents a one-variable polynomial, defined by two-dimensional coefficient array.
    """

    __slots__ = ('_coefs', )
    _fields = ('Coefs', 'order1', 'order2')
    _required = ('Coefs', )
    _numeric_format = {'Coefs': FLOAT_FORMAT}

    def __init__(self, Coefs=None, **kwargs):
        """
        Parameters
        ----------
        Coefs : numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self._coefs = None
        self.Coefs = Coefs
        super(Poly2DType, self).__init__(**kwargs)

    def __call__(self, x, y):
        """
        Evaluate a polynomial at points [`x`, `y`]. This passes `x`,`y` straight through to :func:`polyval2d` of
        `numpy.polynomial.polynomial`.

        Parameters
        ----------
        x : float|int|numpy.ndarray
            The first dependent variable of point(s) at which to evaluate.
        y : float|int|numpy.ndarray
            The second dependent variable of point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        return numpy.polynomial.polynomial.polyval2d(x, y, self._coefs)

    @property
    def order1(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent1 presented in the monomial terms of coefs.
        """

        return self._coefs.shape[0] - 1

    @property
    def order2(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent2 presented in the monomial terms of coefs.
        """

        return self._coefs.shape[1] - 1

    @property
    def Coefs(self):
        """
        numpy.ndarray: The two-dimensional polynomial coefficient array of dtype=float64. Assignment object must be a
        two-dimensional numpy.ndarray, or naively convertible to one.

        .. Note:: this returns the direct coefficient array. Use the `get_array()` method to get a copy of the
            coefficient array of specified data type.
        """

        return self._coefs

    @Coefs.setter
    def Coefs(self, value):
        if value is None:
            raise ValueError('The coefficient array for a Poly2DType instance must be defined.')

        if isinstance(value, (list, tuple)):
            value = numpy.array(value, dtype=numpy.float64)

        if not isinstance(value, numpy.ndarray):
            raise ValueError(
                'Coefs for class Poly2D must be a list or numpy.ndarray. Received type {}.'.format(type(value)))
        elif len(value.shape) != 2:
            raise ValueError(
                'Coefs for class Poly2D must be two-dimensional. Received numpy.ndarray '
                'of shape {}.'.format(value.shape))
        elif not value.dtype.name == 'float64':
            value = numpy.cast[numpy.float64](value)
        self._coefs = value

    def __getitem__(self, item):
        return self._coefs[item]

    def __setitem__(self, item, value):
        self._coefs[item] = value

    def shift(self, t1_shift=0, t1_scale=1, t2_shift=0, t2_scale=1, return_poly=False):
        r"""
        Transform a polynomial with respect to a affine shift in the coordinate system.
        That is, :math:`P(x1, x2) = Q(t1_scale\cdot(t1 - t1_shift), t2_scale\cdot(t2 - t2_shift))`.

        Be careful to follow the convention that the transformation parameters express the
        *current coordinate system* as a shifted, **and then** scaled version of the
        *new coordinate system*.

        Parameters
        ----------
        t1_shift : float
            the **current center coordinate** in the **new coordinate system.**
            That is, `x1=0` when `t1=t1_shift`.
        t1_scale : float
            the scale. That is, when `t1 = t1_shift + 1`, then `x1 = t1_scale`.
            **NOTE:** it is assumed that the coordinate system is re-centered, and **then** scaled.
        t2_shift : float
            the **current center coordinate** in the **new coordinate system.**
            That is, `x2=0` when `t2=t2_shift`.
        t2_scale : float
            the scale. That is, when `t2 = t2_shift + 1`, then `x2 = t2_scale`.
            **NOTE:** it is assumed that the coordinate system is re-centered, and **then** scaled.
        return_poly : bool
            if `True`, a Poly2DType object be returned, otherwise the coefficients array is returned.

        Returns
        -------
        Poly2DType|numpy.ndarray
        """
        # prepare our array workspace
        out = numpy.copy(self._coefs)

        # handle first axis - everything is commutative, so order doesn't matter
        if t1_shift != 0 and self._coefs.shape[0] > 1:
            siz = out.shape[0]
            # let's use horner's method, so iterate from top down
            for i in range(siz):
                index = siz-i-1
                if i > 0:
                    out[index:siz-1, :] -= t1_shift*out[index+1:siz, :]
        if t1_scale != 1 and out.shape[0] > 1:
            out = numpy.power(t1_scale, numpy.arange(out.shape[0]))[:, numpy.newaxis]*out

        # handle second axis
        if t2_shift != 0 and out.shape[1] > 1:
            siz = out.shape[1]
            # let's use horner's method, so iterate from top down
            for i in range(siz):
                index = siz-i-1
                if i > 0:
                    out[:, index:siz-1] -= t2_shift*out[:, index+1:siz]
        if t2_scale != 1 and out.shape[1] > 1:
            out *= numpy.power(t2_scale, numpy.arange(out.shape[1]))

        if return_poly:
            return Poly2DType(Coefs=out)
        else:
            return out

    @classmethod
    def from_array(cls, array):
        """
        Create from the coefficients array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            must be two-dimensional.

        Returns
        -------
        Poly2DType
        """
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
        order1 = int(node.attrib['order1'])
        order2 = int(node.attrib['order2'])
        coefs = numpy.zeros((order1+1, order2+1), dtype=numpy.float64)

        coef_key = cls._child_xml_ns_key.get('Coefs', ns_key)
        coef_nodes = find_children(node, 'Coef', xml_ns, coef_key)
        for cnode in coef_nodes:
            ind1 = int(cnode.attrib['exponent1'])
            ind2 = int(cnode.attrib['exponent2'])
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

        node.attrib['order1'] = str(self.order1)
        node.attrib['order2'] = str(self.order2)
        fmt_func = self._get_formatter('Coefs')
        for i, val1 in enumerate(self._coefs):
            for j, val in enumerate(val1):
                # if val != 0.0:  # should we serialize it sparsely?
                cnode = create_text_node(doc, ctag, fmt_func(val), parent=node)
                cnode.attrib['exponent1'] = str(i)
                cnode.attrib['exponent2'] = str(j)
        return node

    def to_dict(self,  check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = OrderedDict()
        out['Coefs'] = self.Coefs.tolist()
        return out

    def minimize_order(self):
        """
        Trim the trailing zeros for the coefficient array. This modifies the object in place.

        Returns
        -------
        None
        """

        boolc = (self.Coefs != 0)
        if not numpy.any(boolc):
            self.Coefs = numpy.zeros((1, 1), dtype='float64')
            return

        col_inds, row_inds = numpy.meshgrid(
            numpy.arange(self.Coefs.shape[1]), numpy.arange(self.Coefs.shape[0]))
        last_row_ind = numpy.amax(row_inds[boolc])
        last_col_ind = numpy.amax(col_inds[boolc])
        if last_row_ind == self.Coefs.shape[0]-1 and last_col_ind == self.Coefs.shape[1]-1:
            return

        if last_row_ind == 0 and last_col_ind == 0:
            self.Coefs = numpy.array([[self.Coefs[0, 0], ], ], dtype='float64')
        elif last_row_ind == 0:
            self.Coefs = numpy.reshape(self.Coefs[0, :last_col_ind+1], (1, -1))
        elif last_col_ind == 0:
            self.Coefs = numpy.reshape(self.Coefs[:last_row_ind+1, 0], (-1, 1))
        else:
            self.Coefs = self.Coefs[:last_row_ind+1, :last_col_ind+1]


class XYZPolyType(Serializable, Arrayable):
    """
    Represents a single variable polynomial for each of `X`, `Y`, and `Z`. This gives position in ECF coordinates
    as a function of a single dependent variable.
    """

    _fields = ('X', 'Y', 'Z')
    _required = _fields
    # descriptors
    X = SerializableDescriptor(
        'X', Poly1DType, _required, strict=True,
        docstring='The polynomial for the X coordinate.')  # type: Poly1DType
    Y = SerializableDescriptor(
        'Y', Poly1DType, _required, strict=True,
        docstring='The polynomial for the Y coordinate.')  # type: Poly1DType
    Z = SerializableDescriptor(
        'Z', Poly1DType, _required, strict=True,
        docstring='The polynomial for the Z coordinate.')  # type: Poly1DType

    def __init__(self, X=None, Y=None, Z=None, **kwargs):
        """
        Parameters
        ----------
        X : Poly1DType|numpy.ndarray|list|tuple
        Y : Poly1DType|numpy.ndarray|list|tuple
        Z : Poly1DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.X, self.Y, self.Z = X, Y, Z
        super(XYZPolyType, self).__init__(**kwargs)

    def __call__(self, t):
        """
        Evaluate the polynomial at points `t`. This passes `t` straight through
        to :func:`polyval` of `numpy.polynomial.polynomial` for each of
        `X,Y,Z` components. If any of `X,Y,Z` is not populated, then None is returned.

        Parameters
        ----------
        t : float|int|numpy.ndarray
            The point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        x = self.X(t)
        y = self.Y(t)
        z = self.Z(t)
        if numpy.ndim(x) == 0:
            return numpy.array([x, y, z], dtype=x.dtype)
        else:
            o_shape = x.shape
            x = numpy.reshape(x, (-1, 1))
            y = numpy.reshape(y, (-1, 1))
            z = numpy.reshape(z, (-1, 1))
            out = numpy.hstack((x, y, z))
            return numpy.reshape(out, o_shape + (3, ))

    def get_array(self, dtype='object'):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return.
            If `object`, an array of Poly1DType objects is returned.
            Otherwise, an ndarray of shape (3, N) of coefficient vectors is returned.

        Returns
        -------
        numpy.ndarray
            array of the form `[X,Y,Z]`.
        """

        if dtype in ['object', numpy.dtype('object')]:
            return numpy.array([self.X, self.Y, self.Z], dtype='object')
        else:
            # return a 3 x N array of coefficients
            xv = self.X.Coefs
            yv = self.Y.Coefs
            zv = self.Z.Coefs
            length = max(xv.size, yv.size, zv.size)
            out = numpy.zeros((3, length), dtype=dtype)
            out[0, :xv.size] = xv
            out[1, :yv.size] = yv
            out[2, :zv.size] = zv
            return out

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed `[X, Y, Z]`

        Returns
        -------
        XYZPolyType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(X=array[0], Y=array[1], Z=array[2])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))

    def derivative(self, der_order=1, return_poly=False):
        """
        Calculate the `der_order` derivative of each component polynomial.

        Parameters
        ----------
        der_order : int
            the order of the derivative
        return_poly : bool
            if `True`, a XYZPolyType if returned, otherwise a list of the coefficient arrays is returned.

        Returns
        -------
        XYZPolyType|list
        """

        coefs = [
            getattr(self, attrib).derivative(der_order=der_order, return_poly=False) for attrib in ['X', 'Y', 'Z']]

        if return_poly:
            return XYZPolyType(X=coefs[0], Y=coefs[1], Z=coefs[2])
        return coefs

    def derivative_eval(self, t, der_order=1):
        """
        Evaluate the `der_order` derivative of the polynomial collection at points `x`.
        This uses the functionality presented in `numpy.polynomial.polynomial`.

        Parameters
        ----------
        t : float|int|numpy.ndarray
            The point(s) at which to evaluate.
        der_order : int
            The derivative.

        Returns
        -------
        numpy.ndarray
        """

        der_poly = self.derivative(der_order=der_order, return_poly=True)
        return der_poly(t)

    def shift(self, t_0, alpha=1, return_poly=False):
        r"""
        Transform a polynomial with respect to a affine shift in the coordinate system.
        That is, :math:`P(u) = Q(\alpha\cdot(t-t_0))`.

        Be careful to follow the convention that the transformation parameters express the *current coordinate system*
        as a shifted, **and then** scaled version of the *new coordinate system*. If the new coordinate is
        :math:`t = \beta\cdot u - t_0`, then :math:`u = (t - t_0)/\beta`, and :math:`\alpha = 1/\beta`.

        Parameters
        ----------
        t_0 : float
            the **current center coordinate** in the **new coordinate system.**
            That is, `u=0` when `t=t_0`.

        alpha : float
            the scale. That is, when `t = t0 + 1`, then :math:`u = \alpha`.

        return_poly : bool
            if `True`, an XYZPolyType instance is returned, otherwise a list of the coefficient arrays is returned.

        Returns
        -------
        XYZPolyType|list
        """

        coefs = [
            getattr(self, attrib).shift(t_0, alpha=alpha, return_poly=False) for attrib in ['X', 'Y', 'Z']]

        if return_poly:
            return XYZPolyType(X=coefs[0], Y=coefs[1], Z=coefs[2])
        return coefs

    def minimize_order(self):
        """
        Trim the trailing zeros for each component coefficient array. This
        modifies the object in place.

        Returns
        -------
        None
        """

        self.X.minimize_order()
        self.Y.minimize_order()
        self.Z.minimize_order()


class XYZPolyAttributeType(XYZPolyType):
    """
    An array element of X, Y, Z polynomials. The output of these polynomials are expected
    to be spatial variables in the ECF coordinate system.
    """
    _fields = ('X', 'Y', 'Z', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = IntegerDescriptor(
        'index', _required, strict=False, docstring='The array index value.')  # type: int

    def __init__(self, X=None, Y=None, Z=None, index=None, **kwargs):
        """
        Parameters
        ----------
        X : Poly1DType|numpy.ndarray|list|tuple
        Y : Poly1DType|numpy.ndarray|list|tuple
        Z : Poly1DType|numpy.ndarray|list|tuple
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.index = index
        super(XYZPolyAttributeType, self).__init__(X=X, Y=Y, Z=Z, **kwargs)

    @classmethod
    def from_array(cls, array, index=1):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [X, Y, Z]
        index : int
            the array index

        Returns
        -------
        XYZPolyAttributeType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(X=array[0], Y=array[1], Z=array[2], index=index)
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class GainPhasePolyType(Serializable):
    """A container for the Gain and Phase Polygon definitions."""

    _fields = ('GainPoly', 'PhasePoly')
    _required = _fields
    # descriptors
    GainPoly = SerializableDescriptor(
        'GainPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='One-way signal gain (in dB) as a function of X-axis direction cosine (DCX) (variable 1) '
                  'and Y-axis direction cosine (DCY) (variable 2). Gain relative to gain at DCX = 0 '
                  'and DCY = 0, so constant coefficient is always 0.0.')  # type: Poly2DType
    PhasePoly = SerializableDescriptor(
        'PhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='One-way signal phase (in cycles) as a function of DCX (variable 1) and '
                  'DCY (variable 2). Phase relative to phase at DCX = 0 and DCY = 0, '
                  'so constant coefficient is always 0.0.')  # type: Poly2DType

    def __init__(self, GainPoly=None, PhasePoly=None, **kwargs):
        """
        Parameters
        ----------
        GainPoly : Poly2DType|numpy.ndarray|list|tuple
        PhasePoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GainPoly = GainPoly
        self.PhasePoly = PhasePoly
        super(GainPhasePolyType, self).__init__(**kwargs)

    def __call__(self, x, y):
        """
        Evaluate a polynomial at points [`x`, `y`]. This passes `x`,`y` straight
        through to the call method for each component.

        Parameters
        ----------
        x : float|int|numpy.ndarray
            The first dependent variable of point(s) at which to evaluate.
        y : float|int|numpy.ndarray
            The second dependent variable of point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        if self.GainPoly is None or self.PhasePoly is None:
            return None
        return numpy.array([self.GainPoly(x, y), self.PhasePoly(x, y)], dtype=numpy.float64)

    def minimize_order(self):
        """
        Trim the trailing zeros for each component coefficient array. This
        modifies the object in place.

        Returns
        -------
        None
        """

        self.GainPoly.minimize_order()
        self.PhasePoly.minimize_order()


#############
# Error Decorrelation type


class ErrorDecorrFuncType(Serializable):
    r"""
    This container allows parameterization of linear error decorrelation rate model.
    If :math:`(\Delta t) = |t2 - t1|`, then

    .. math::

        CC(\Delta t) = \min(1.0, \max(0.0, CC0 - DCR\cdot(\Delta t)))
    """

    _fields = ('CorrCoefZero', 'DecorrRate')
    _required = _fields
    _numeric_format = {'CorrCoefZero': FLOAT_FORMAT, 'DecorrRate': FLOAT_FORMAT}
    # descriptors
    CorrCoefZero = FloatDescriptor(
        'CorrCoefZero', _required, strict=True, bounds=(-1, 1),
        docstring='Error correlation coefficient for zero time difference (CC0).')  # type: float
    DecorrRate = FloatDescriptor(
        'DecorrRate', _required, strict=True, bounds=(0, None),
        docstring='Error decorrelation rate. Simple linear decorrelation rate (DCR).')  # type: float

    def __init__(self, CorrCoefZero=None, DecorrRate=None, **kwargs):
        """
        Parameters
        ----------
        CorrCoefZero : float
        DecorrRate : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CorrCoefZero = CorrCoefZero
        self.DecorrRate = DecorrRate
        super(ErrorDecorrFuncType, self).__init__(**kwargs)
