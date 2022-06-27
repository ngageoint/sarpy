"""
The Per Vector parameters (PVP) definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List, Tuple, Optional

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


class PerVectorParameterEB(Serializable):
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
        super(PerVectorParameterEB, self).__init__(**kwargs)

    @property
    def Size(self):
        """
        int: The size of the vector, constant value 2 here.
        """

        return 2

    @property
    def Format(self):
        """
        str: The format of the vector data, constant value 'DCX=F8;DCY=F8;' here.
        """

        return 'DCX=F8;DCY=F8;'


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


class TxAntennaType(Serializable):
    _fields = ('TxACX', 'TxACY', 'TxEB')
    _required = _fields
    TxACX = SerializableDescriptor(
        'TxACX', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    TxACY = SerializableDescriptor(
        'TxACY', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    TxEB = SerializableDescriptor(
        'TxEB', PerVectorParameterEB, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterEB

    def __init__(
            self,
            TxACX: PerVectorParameterXYZ = None,
            TxACY: PerVectorParameterXYZ = None,
            TxEB: PerVectorParameterEB = None,
            **kwargs):
        """
        Parameters
        ----------
        TxACX : PerVectorParameterXYZ
        TxACY : PerVectorParameterXYZ
        TxEB : PerVectorParameterEB
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxACX = TxACX
        self.TxACY = TxACY
        self.TxEB = TxEB
        super(TxAntennaType, self).__init__(**kwargs)


class RcvAntennaType(Serializable):
    _fields = ('RcvACX', 'RcvACY', 'RcvEB')
    _required = _fields
    RcvACX = SerializableDescriptor(
        'RcvACX', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvACY = SerializableDescriptor(
        'RcvACY', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvEB = SerializableDescriptor(
        'RcvEB', PerVectorParameterEB, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterEB

    def __init__(
            self,
            RcvACX: Optional[PerVectorParameterXYZ] = None,
            RcvACY: Optional[PerVectorParameterXYZ] = None,
            RcvEB: Optional[PerVectorParameterEB] = None,
            **kwargs):
        """
        Parameters
        ----------
        RcvACX : PerVectorParameterXYZ
        RcvACY : PerVectorParameterXYZ
        RcvEB : PerVectorParameterEB
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RcvACX = RcvACX
        self.RcvACY = RcvACY
        self.RcvEB = RcvEB
        super(RcvAntennaType, self).__init__(**kwargs)


class PVPType(Serializable):
    _fields = (
        'TxTime', 'TxPos', 'TxVel', 'RcvTime', 'RcvPos', 'RcvVel',
        'SRPPos', 'AmpSF', 'aFDOP', 'aFRR1', 'aFRR2', 'FX1', 'FX2',
        'FXN1', 'FXN2', 'TOA1', 'TOA2', 'TOAE1', 'TOAE2', 'TDTropoSRP',
        'TDIonoSRP', 'SC0', 'SCSS', 'SIGNAL',
        'TxAntenna', 'RcvAntenna', 'AddedPVP')
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
    TxAntenna = SerializableDescriptor(
        'TxAntenna', TxAntennaType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[TxAntennaType]
    RcvAntenna = SerializableDescriptor(
        'RcvAntenna', RcvAntennaType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[RcvAntennaType]
    AddedPVP = SerializableListDescriptor(
        'AddedPVP', UserDefinedPVPType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, List[UserDefinedPVPType]]

    def __init__(
            self,
            TxTime: PerVectorParameterF8 = None,
            TxPos: PerVectorParameterXYZ = None,
            TxVel: PerVectorParameterXYZ = None,
            RcvTime: PerVectorParameterF8 = None,
            RcvPos: PerVectorParameterXYZ = None,
            RcvVel: PerVectorParameterXYZ = None,
            SRPPos: PerVectorParameterXYZ = None,
            AmpSF: Optional[PerVectorParameterF8] = None,
            aFDOP: PerVectorParameterF8 = None,
            aFRR1: PerVectorParameterF8 = None,
            aFRR2: PerVectorParameterF8 = None,
            FX1: PerVectorParameterF8 = None,
            FX2: PerVectorParameterF8 = None,
            FXN1: Optional[PerVectorParameterF8] = None,
            FXN2: Optional[PerVectorParameterF8] = None,
            TOA1: PerVectorParameterF8 = None,
            TOA2: PerVectorParameterF8 = None,
            TOAE1: Optional[PerVectorParameterF8] = None,
            TOAE2: Optional[PerVectorParameterF8] = None,
            TDTropoSRP: PerVectorParameterF8 = None,
            TDIonoSRP: Optional[PerVectorParameterF8] = None,
            SC0: PerVectorParameterF8 = None,
            SCSS: PerVectorParameterF8 = None,
            SIGNAL: Optional[PerVectorParameterI8] = None,
            TxAntenna: Optional[TxAntennaType] = None,
            RcvAntenna: Optional[RcvAntennaType] = None,
            AddedPVP: Optional[List[UserDefinedPVPType]] = None,
            **kwargs):
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
        TxAntenna : None|TxAntennaType
        RcvAntenna : None|RcvAntennaType
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
        self.TxAntenna = TxAntenna
        self.RcvAntenna = RcvAntenna
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
        for fld in self._fields[:-3]:
            val = getattr(self, fld)
            if val is not None:
                out += val.Size*8
        for fld in ['TxAntenna', 'RcvAntenna']:
            val = getattr(self, fld)
            assert(isinstance(val, TxAntennaType))
            if val is not None:
                out += (3 + 3 + 2)*8
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
        None|Tuple[int, int, str]
        """

        def get_return(the_val) -> Union[None, Tuple[int, int, str]]:
            if the_val is None:
                return None
            return the_val.Offset*8, the_val.Size*8, homogeneous_dtype(the_val.Format).char

        if field in self._fields[:-3]:
            return get_return(getattr(self, field))
        elif field in ['TxACX', 'TxACY', 'TxEB']:
            if self.TxAntenna is None:
                return None
            else:
                return get_return(getattr(self.TxAntenna, field))
        elif field in ['RcvACX', 'RcvACY', 'RcvEB']:
            if self.RcvAntenna is None:
                return None
            else:
                return get_return(getattr(self.RcvAntenna, field))
        else:
            if self.AddedPVP is None:
                return None
            for val in self.AddedPVP:
                if field == val.Name:
                    return get_return(val)
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
            elif fld == 'TxAntenna':
                for t_fld in ['TxACX', 'TxACY', 'TxEB']:
                    t_val = getattr(val, t_fld)
                    names.append(t_fld)
                    formats.append(binary_format_string_to_dtype(t_val.Format))
                    offsets.append(t_val.Offset*bytes_per_word)
            elif fld == 'RcvAntenna':
                for t_fld in ['RcvACX', 'RcvACY', 'RcvEB']:
                    t_val = getattr(val, t_fld)
                    names.append(t_fld)
                    formats.append(binary_format_string_to_dtype(t_val.Format))
                    offsets.append(t_val.Offset*bytes_per_word)
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

    def version_required(self) -> Tuple[int, int, int]:
        required = (1, 0, 1)
        if self.TxAntenna is not None or self.RcvAntenna is not None:
            required = max(required, (1, 1, 0))
        return required
