"""
The Per Vector parameters (PVP) definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List

import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import StringDescriptor, IntegerDescriptor, \
    SerializableDescriptor, SerializableListDescriptor
from .utils import binary_format_string_to_dtype, homogeneous_dtype
from .base import DEFAULT_STRICT


class PerVectorParameterI8(Serializable):
    _fields = ('Offset', 'Size', 'Format')
    _required = ('Offset', )
    # descriptors
    Offset = IntegerDescriptor(
        'Offset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The offset value.')  # type: int

    def __init__(self, Offset=None, **kwargs):
        """

        Parameters
        ----------
        Offset : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Offset = Offset
        super(PerVectorParameterI8, self).__init__(**kwargs)

    @property
    def Size(self):
        """
        int: The size of the vector, constant value 1 here.
        """

        return 1

    @property
    def Format(self):
        """
        str: The format of the vector data, constant value 'I8' here.
        """

        return 'I8'


class PerVectorParameterF8(Serializable):
    _fields = ('Offset', 'Size', 'Format')
    _required = ('Offset', )
    # descriptors
    Offset = IntegerDescriptor(
        'Offset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The offset value.')  # type: int

    def __init__(self, Offset=None, **kwargs):
        """

        Parameters
        ----------
        Offset : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Offset = Offset
        super(PerVectorParameterF8, self).__init__(**kwargs)

    @property
    def Size(self):
        """
        int: The size of the vector, constant value 1 here.
        """

        return 1

    @property
    def Format(self):
        """
        str: The format of the vector data, constant value 'F8' here.
        """

        return 'F8'


class PerVectorParameterXYZ(Serializable):
    _fields = ('Offset', 'Size', 'Format')
    _required = ('Offset', )
    # descriptors
    Offset = IntegerDescriptor(
        'Offset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The offset value.')  # type: int

    def __init__(self, Offset=None, **kwargs):
        """

        Parameters
        ----------
        Offset : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Offset = Offset
        super(PerVectorParameterXYZ, self).__init__(**kwargs)

    @property
    def Size(self):
        """
        int: The size of the vector, constant value 3 here.
        """

        return 3

    @property
    def Format(self):
        """
        str: The format of the vector data, constant value 'X=F8;Y=F8;Z=F8;' here.
        """

        return 'X=F8;Y=F8;Z=F8;'


class UserDefinedPVPType(Serializable):
    """
    A user defined PVP structure.
    """

    _fields = ('Name', 'Offset', 'Size', 'Format')
    _required = _fields
    # descriptors
    Name = StringDescriptor(
        'Name', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    Offset = IntegerDescriptor(
        'Offset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int
    Size = IntegerDescriptor(
        'Size', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    Format = StringDescriptor(
        'Format', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, Name=None, Offset=None, Size=None, Format=None, **kwargs):
        """

        Parameters
        ----------
        Name : str
        Offset : int
        Size : int
        Format : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.Offset = Offset
        self.Size = Size
        self.Format = Format
        super(UserDefinedPVPType, self).__init__(**kwargs)


class PVPType(Serializable):
    _fields = (
        'TxTime', 'TxPos', 'TxVel', 'RcvTime', 'RcvPos', 'RcvVel',
        'SRPPos', 'AmpSF', 'aFDOP', 'aFRR1', 'aFRR2', 'FX1', 'FX2',
        'FXN1', 'FXN2', 'TOA1', 'TOA2', 'TOAE1', 'TOAE2', 'TDTropoSRP',
        'TDIonoSRP', 'SC0', 'SCSS', 'SIGNAL', 'AddedPVP')
    _required = (
        'TxTime', 'TxPos', 'TxVel', 'RcvTime', 'RcvPos', 'RcvVel',
        'SRPPos', 'aFDOP', 'aFRR1', 'aFRR2', 'FX1', 'FX2',
        'TOA1', 'TOA2', 'TDTropoSRP', 'SC0', 'SCSS')
    _collections_tags = {'AddedPVP': {'array': False, 'child_tag': 'AddedPVP'}}
    # descriptors
    TxTime = SerializableDescriptor(
        'TxTime', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TxPos = SerializableDescriptor(
        'TxPos', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    TxVel = SerializableDescriptor(
        'TxVel', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvTime = SerializableDescriptor(
        'RcvTime', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    RcvPos = SerializableDescriptor(
        'RcvPos', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvVel = SerializableDescriptor(
        'RcvVel', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    SRPPos = SerializableDescriptor(
        'SRPPos', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    AmpSF = SerializableDescriptor(
        'AmpSF', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    aFDOP = SerializableDescriptor(
        'aFDOP', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    aFRR1 = SerializableDescriptor(
        'aFRR1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    aFRR2 = SerializableDescriptor(
        'aFRR2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FX1 = SerializableDescriptor(
        'FX1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FX2 = SerializableDescriptor(
        'FX2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FXN1 = SerializableDescriptor(
        'FXN1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FXN2 = SerializableDescriptor(
        'FXN2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TOA1 = SerializableDescriptor(
        'TOA1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TOA2 = SerializableDescriptor(
        'TOA2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TOAE1 = SerializableDescriptor(
        'TOAE1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TOAE2 = SerializableDescriptor(
        'TOAE2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TDTropoSRP = SerializableDescriptor(
        'TDTropoSRP', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TDIonoSRP = SerializableDescriptor(
        'TDIonoSRP', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    SC0 = SerializableDescriptor(
        'SC0', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    SCSS = SerializableDescriptor(
        'SCSS', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    SIGNAL = SerializableDescriptor(
        'SIGNAL', PerVectorParameterI8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterI8
    AddedPVP = SerializableListDescriptor(
        'AddedPVP', UserDefinedPVPType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, List[UserDefinedPVPType]]

    def __init__(self, TxTime=None, TxPos=None, TxVel=None,
                 RcvTime=None, RcvPos=None, RcvVel=None,
                 SRPPos=None, AmpSF=None,
                 aFDOP=None, aFRR1=None, aFRR2=None,
                 FX1=None, FX2=None, FXN1=None, FXN2=None,
                 TOA1=None, TOA2=None, TOAE1=None, TOAE2=None,
                 TDTropoSRP=None, TDIonoSRP=None, SC0=None, SCSS=None,
                 SIGNAL=None, AddedPVP=None, **kwargs):
        """

        Parameters
        ----------
        TxTime : PerVectorParameterF8
        TxPos : PerVectorParameterXYZ
        TxVel : PerVectorParameterXYZ
        RcvTime : PerVectorParameterF8
        RcvPos : PerVectorParameterXYZ
        RcvVel : PerVectorParameterXYZ
        SRPPos : PerVectorParameterXYZ
        AmpSF : None|PerVectorParameterF8
        aFDOP : PerVectorParameterF8
        aFRR1 : PerVectorParameterF8
        aFRR2 : PerVectorParameterF8
        FX1 : PerVectorParameterF8
        FX2 : PerVectorParameterF8
        FXN1 : None|PerVectorParameterF8
        FXN2 : None|PerVectorParameterF8
        TOA1 : PerVectorParameterF8
        TOA2 : PerVectorParameterF8
        TOAE1 : None|PerVectorParameterF8
        TOAE2 : None|PerVectorParameterF8
        TDTropoSRP : PerVectorParameterF8
        TDIonoSRP : None|PerVectorParameterF8
        SC0 : PerVectorParameterF8
        SCSS : PerVectorParameterF8
        SIGNAL : None|PerVectorParameterI8
        AddedPVP : None|List[UserDefinedPVPType]
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxTime = TxTime
        self.TxPos = TxPos
        self.TxVel = TxVel
        self.RcvTime = RcvTime
        self.RcvPos = RcvPos
        self.RcvVel = RcvVel
        self.SRPPos = SRPPos
        self.AmpSF = AmpSF
        self.aFDOP = aFDOP
        self.aFRR1 = aFRR1
        self.aFRR2 = aFRR2
        self.FX1 = FX1
        self.FX2 = FX2
        self.FXN1 = FXN1
        self.FXN2 = FXN2
        self.TOA1 = TOA1
        self.TOA2 = TOA2
        self.TOAE1 = TOAE1
        self.TOAE2 = TOAE2
        self.TDTropoSRP = TDTropoSRP
        self.TDIonoSRP = TDIonoSRP
        self.SC0 = SC0
        self.SCSS = SCSS
        self.SIGNAL = SIGNAL
        self.AddedPVP = AddedPVP
        super(PVPType, self).__init__(**kwargs)

    def get_size(self):
        """
        Gets the size in bytes of each vector.

        Returns
        -------
        int
        """

        out = 0
        for fld in self._fields[:-1]:
            val = getattr(self, fld)
            if val is not None:
                out += val.Size*8
        if self.AddedPVP is not None:
            for entry in self.AddedPVP:
                out += entry.Size*8
        return out

    def get_offset_size_format(self, field):
        """
        Get the Offset (in bytes), Size (in bytes) for the given field,
        as well as the corresponding struct format string.

        Parameters
        ----------
        field : str
            The desired field name.

        Returns
        -------
        None|(int, int, str)
        """

        if field in self._fields[:-1]:
            val = getattr(self, field)
            if val is None:
                return None
            return val.Offset*8, val.Size*8, homogeneous_dtype(val.Format).char
        else:
            if self.AddedPVP is None:
                return None
            for val in self.AddedPVP:
                if field == val.Name:
                    return val.Offset*8, val.Size*8, homogeneous_dtype(val.Format).char
            return None

    def get_vector_dtype(self):
        """
        Gets the dtype for the corresponding structured array for the full PVP
        array.

        Returns
        -------
        numpy.dtype
            This will be a compound dtype for a structured array.
        """

        bytes_per_word = 8
        names = []
        formats = []
        offsets = []

        for fld in self._fields:
            val = getattr(self, fld)
            if val is None:
                continue
            elif fld == 'AddedPVP':
                for entry in val:
                    assert isinstance(entry, UserDefinedPVPType)
                    names.append(entry.Name)
                    formats.append(binary_format_string_to_dtype(entry.Format))
                    offsets.append(entry.Offset*bytes_per_word)
            else:
                names.append(fld)
                formats.append(binary_format_string_to_dtype(val.Format))
                offsets.append(val.Offset*bytes_per_word)
        return numpy.dtype({'names': names, 'formats': formats, 'offsets': offsets})
