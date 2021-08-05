"""
Common use elements for the AFRL labeling definition
"""

import numpy

# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import _DateTimeDescriptor, \
    _FloatDescriptor, _StringDescriptor, Arrayable, Serializable


class DateRangeType(Serializable):
    """
    A range of dates with resolution of 1 day
    """
    _fields = ('Begin', 'End')
    _required = _fields
    # descriptors
    Begin = _DateTimeDescriptor(
        'Begin', _required, strict=True, numpy_datetime_units='D',
        docstring="Begin date of the data collection.")
    End = _DateTimeDescriptor(
        'End', _required, strict=True, numpy_datetime_units='D',
        docstring="End date of the data collection.")

    def __init__(self, Begin=None, End=None, **kwargs):
        """
        Parameters
        ----------
        Begin : str
        End : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Begin, self.End = Begin, End
        super(DateRangeType, self).__init__(**kwargs)

    def get_dates(self):
        """
        Gets the begin and end dates of the class instance.

        Returns
        -------
        tuple
            tuple of the form (Begin, End), both numpy.datetime64
        """

        return (self.Begin, self.End)


class LatLonWithNameType(Serializable):
    _fields = ('Lat', 'Lon', 'Name')
    _required = _fields
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', _required, strict=True,
        docstring="General latitude of the data collection.")
    Lon = _FloatDescriptor(
        'Lon', _required, strict=True,
        docstring="General longitude of the data collection.")
    Name = _StringDescriptor(
        'Name', _required,
        docstring="Common name of the collection location.")

    def __init__(self, Lat=None, Lon=None, Name=None, **kwargs):
        """
        Parameters
        ----------
        Lat : None|float
        Lon : None|float
        Name : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Lat = Lat
        self.Lon = Lon
        self.Name = Name
        super(LatLonWithNameType, self).__init__(**kwargs)

    def get_lat_lon_name(self):
        """
        Gets lat/lon and name from the class instance.

        Returns
        -------
        tuple
            tuple of the form (Lat, Lon, Name) where Lat and Lon are floats
            and Name is str
        """


class RangeCrossRangeType(Serializable, Arrayable):
    """
    A range and cross range attribute container
    """
    _fields = ('Range', 'CrossRange')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # descriptors
    Range = _FloatDescriptor(
        'Range', _required, strict=True, docstring='The Range attribute.')  # type: float
    CrossRange = _FloatDescriptor(
        'CrossRange', _required, strict=True, docstring='The Cross Range attribute.')  # type: float

    def __init__(self, Range=None, CrossRange=None, **kwargs):
        """
        Parameters
        ----------
        Range : float
        CrossRange : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Range, self.CrossRange = Range, CrossRange
        super(RangeCrossRangeType, self).__init__(**kwargs)

    def get_array(self, dtype='float64'):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Range, CrossRange]
        """

        return numpy.array([self.Range, self.CrossRange], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Range, CrossRange]

        Returns
        -------
        RangeCrossRangeType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Range=array[0], CrossRange=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class RowColDoubleType(Serializable, Arrayable):
    _fields = ('Row', 'Col')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # Descriptors
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
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Row, self.Col = Row, Col
        super(RowColDoubleType, self).__init__(**kwargs)

    def get_array(self, dtype='float64'):
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
