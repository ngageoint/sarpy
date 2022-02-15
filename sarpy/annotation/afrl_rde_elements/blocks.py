"""
Common use elements for the NGA modified RDE/AFRL labeling definition
"""

__classification__ = "UNCLASSIFIED"
__authors__ = "Thomas McCullough"


import numpy
from datetime import date, datetime
from typing import Optional

from sarpy.io.xml.base import Serializable, Arrayable
from sarpy.io.xml.descriptors import DateTimeDescriptor, FloatDescriptor, \
    SerializableDescriptor, StringEnumDescriptor, StringDescriptor

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.complex.sicd_elements.blocks import XYZType
from .base import DEFAULT_STRICT


class DateRangeType(Serializable, Arrayable):
    """
    A range of dates with resolution of 1 day
    """
    _fields = ('Begin', 'End')
    _required = _fields
    # descriptors
    Begin = DateTimeDescriptor(
        'Begin', _required, strict=DEFAULT_STRICT, numpy_datetime_units='D',
        docstring="Begin date of the data collection.")  # type: Optional[numpy.datetime64]
    End = DateTimeDescriptor(
        'End', _required, strict=DEFAULT_STRICT, numpy_datetime_units='D',
        docstring="End date of the data collection.")  # type: Optional[numpy.datetime64]

    def __init__(self, Begin=None, End=None, **kwargs):
        """
        Parameters
        ----------
        Begin : None|numpy.datetime64|str|datetime|date
        End : None|numpy.datetime64|str|datetime|date
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Begin = Begin
        self.End = End
        super(DateRangeType, self).__init__(**kwargs)

    def get_array(self, dtype='datetime64[D]'):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array
        """

        return numpy.array([self.Begin, self.End], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Begin, End]

        Returns
        -------
        DateRangeType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Begin=array[0], End=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class DateTimeRangeType(Serializable, Arrayable):
    """
    A range of dates with resolution of 1 day
    """
    _fields = ('Begin', 'End')
    _required = _fields
    # descriptors
    Begin = DateTimeDescriptor(
        'Begin', _required, strict=DEFAULT_STRICT, numpy_datetime_units='s',
        docstring="Begin date/time of the data collection.")  # type: Optional[numpy.datetime64]
    End = DateTimeDescriptor(
        'End', _required, strict=DEFAULT_STRICT, numpy_datetime_units='s',
        docstring="End date/time of the data collection.")  # type: Optional[numpy.datetime64]

    def __init__(self, Begin=None, End=None, **kwargs):
        """
        Parameters
        ----------
        Begin : None|numpy.datetime64|str|datetime|date
        End : None|numpy.datetime64|str|datetime|date
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Begin = Begin
        self.End = End
        super(DateTimeRangeType, self).__init__(**kwargs)

    def get_array(self, dtype='datetime64[D]'):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array
        """

        return numpy.array([self.Begin, self.End], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Begin, End]

        Returns
        -------
        DateTimeRangeType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Begin=array[0], End=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class RangeCrossRangeType(Serializable, Arrayable):
    """
    A range and cross range attribute container
    """
    _fields = ('Range', 'CrossRange')
    _required = _fields
    _numeric_format = {key: '0.17G' for key in _fields}
    # descriptors
    Range = FloatDescriptor(
        'Range', _required, strict=True, docstring='The Range attribute.')  # type: float
    CrossRange = FloatDescriptor(
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
    _numeric_format = {key: '0.17G' for key in _fields}
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


class LatLonEleType(Serializable, Arrayable):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon', 'Ele')
    _required = _fields
    _numeric_format = {'Lat': '0.17G', 'Lon': '0.17G', 'Ele': '0.17G'}
    # descriptors
    Lat = FloatDescriptor(
        'Lat', _required, strict=True,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.'
    )  # type: float
    Lon = FloatDescriptor(
        'Lon', _required, strict=True,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.'
    )  # type: float
    Ele = FloatDescriptor(
        'Ele', _required, strict=True,
        docstring='The Height Above Ellipsoid (in meters) attribute. '
                  'Assumed to be WGS-84 coordinates.')  # type: float

    def __init__(self, Lat=None, Lon=None, Ele=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        Ele : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Lat = Lat
        self.Lon = Lon
        self.Ele = Ele
        super(LatLonEleType, self).__init__(Lat=Lat, Lon=Lon, **kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the data.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        return numpy.array([self.Lat, self.Lon, self.Ele], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon, Ele]

        Returns
        -------
        LatLonEleType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], Ele=array[2])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class ProjectionPerturbationType(Serializable):
    """
    Basic information required for SICD/SIDD projection model perturbation.
    """
    
    _fields = ('CoordinateFrame', 'DeltaArp', 'DeltaVarp', 'DeltaRange')
    _required=('CoordinateFrame', )
    _numeric_format = {'Lat': '0.17G', }
    CoordinateFrame = StringEnumDescriptor(
        'CoordinateFrame', {'ECF', 'RIC_ECI', 'RIC_ECF'}, _required)  # type: str
    DeltaArp = SerializableDescriptor(
        'DeltaArp', XYZType, _required)  # type: XYZType
    DeltaVarp = SerializableDescriptor(
        'DeltaVarp', XYZType, _required)  # type: XYZType
    DeltaRange = FloatDescriptor(
        'DeltaRange', _required)  # type: float

    def __init__(self, CoordinateFrame=None, DeltaArp=None, DeltaVarp=None, DeltaRange=None, **kwargs):
        """
        Parameters
        ----------
        CoordinateFrame : str
        DeltaArp : None|XYZType|numpy.ndarray|list|tuple
        DeltaVarp : None|XYZType|numpy.ndarray|list|tuple
        DeltaRange : None|float
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']

        self.CoordinateFrame = CoordinateFrame
        self.DeltaArp = DeltaArp
        self.DeltaVarp = DeltaVarp
        self.DeltaRange = DeltaRange

        super(ProjectionPerturbationType, self).__init__(**kwargs)

    def set_coa_projection(self, structure):
        """
        Sets the sicd or sidd coa_projection property, as appropriate.

        Parameters
        ----------
        structure : SICDType|SIDDType1|SIDDType2
        """

        if not isinstance(structure, (SICDType, SIDDType1, SIDDType2)):
            raise TypeError('Requires input of type SICDType or SIDDType, got {}'.format(type(structure)))

        structure.define_coa_projection(
            delta_arp=None if self.DeltaArp is None else self.DeltaArp.get_array(dtype='float64'),
            delta_varp=None if self.DeltaVarp is None else self.DeltaVarp.get_array(dtype='float64'),
            range_bias=self.DeltaRange,
            adj_params_frame=self.CoordinateFrame,
            overide=True)


class LabelSourceType(Serializable):
    _fields = ('SourceType', 'SourceID', 'Description')
    _required = ('SourceType', )
    SourceType = StringEnumDescriptor(
        'SourceType', {
            'Ground Truth', 'Analyst Truth', 'Algorithm Truth', 'Other', 'Unknown'},
        _required,
        docstring='The source type of the labeling effort')  # type: str
    SourceID = StringDescriptor(
        'SourceID', _required,
        docstring='The "ID" of the labeling source. '
                  'This should be populated following program guidance.')  # type: Optional[str]
    Description = StringDescriptor(
        'Description', _required,
        docstring='A description of the labeling source')  # type: Optional[str]

    def __init__(self, SourceType='Unknown', SourceID=None, Description=None, **kwargs):
        """
        Parameters
        ----------
        SourceType : str
        SourceID : None|str
        Description : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']

        self.SourceType = SourceType
        self.SourceID = SourceID
        self.Description = Description
        super(LabelSourceType, self).__init__(**kwargs)
