"""
The Per Vector parameters (PVP) definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")

from typing import Union, List

import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import IntegerDescriptor, SerializableDescriptor, \
    SerializableListDescriptor
from sarpy.io.phase_history.cphd1_elements.PVP import PerVectorParameterI8, \
    PerVectorParameterF8, PerVectorParameterXYZ, UserDefinedPVPType
from sarpy.io.phase_history.cphd1_elements.utils import binary_format_string_to_dtype, homogeneous_dtype
from .base import DEFAULT_STRICT


BYTES_PER_WORD = 8


class PerVectorParameterDCXY(Serializable):
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
        super(PerVectorParameterDCXY, self).__init__(**kwargs)

    @property
    def Size(self):
        """
        int: The size of the vector
        """

        return 2

    @property
    def Format(self):
        """
        str: The format of the vector data, constant value 'DCX=F8;DCY=F8;' here.
        """

        return 'DCX=F8;DCY=F8;'


class PerVectorParameterTxLFM(Serializable):
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
        super(PerVectorParameterTxLFM, self).__init__(**kwargs)

    @property
    def Size(self):
        """
        int: The size of the vector, constant value 3 here.
        """

        return 3

    @property
    def Format(self):
        """
        str: The format of the vector data, constant value 'PhiXC=F8;FxC=F8;FxRate=F8;' here.
        """

        return 'PhiXC=F8;FxC=F8;FxRate=F8;'


class TxAntennaType(Serializable):
    _fields = ('TxACX', 'TxACY', 'TxEB')
    _required = _fields
    # descriptors
    TxACX = SerializableDescriptor(
        'TxACX', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    TxACY = SerializableDescriptor(
        'TxACY', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    TxEB = SerializableDescriptor(
        'TxEB', PerVectorParameterDCXY, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterDCXY

    def __init__(self, TxACX=None, TxACY=None, TxEB=None, **kwargs):
        """

        Parameters
        ----------
        TxACX : PerVectorParameterXYZ
        TxACY : PerVectorParameterXYZ
        TxEB : PerVectorParameterDCXY
        kwargs
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
    # descriptors
    RcvACX = SerializableDescriptor(
        'RcvACX', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvACY = SerializableDescriptor(
        'RcvACY', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvEB = SerializableDescriptor(
        'RcvEB', PerVectorParameterDCXY, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterDCXY

    def __init__(self, RcvACX=None, RcvACY=None, RcvEB=None, **kwargs):
        """

        Parameters
        ----------
        RcvACX : PerVectorParameterXYZ
        RcvACY : PerVectorParameterXYZ
        RcvEB : PerVectorParameterDCXY
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RcvACX = RcvACX
        self.RcvACY = RcvACY
        self.RcvEB = RcvEB
        super(RcvAntennaType, self).__init__(**kwargs)


class TxPulseType(Serializable):
    _fields = ('TxTime', 'TxPos', 'TxVel', 'FX1', 'FX2', 'TXmt', 'TxLFM', 'TxAntenna')
    _required = ('TxTime', 'TxPos', 'TxVel', 'FX1', 'FX2', 'TXmt')
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
    FX1 = SerializableDescriptor(
        'FX1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FX2 = SerializableDescriptor(
        'FX2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TXmt = SerializableDescriptor(
        'TXmt', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    TxLFM = SerializableDescriptor(
        'TxLFM', PerVectorParameterTxLFM, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, PerVectorParameterTxLFM]
    TxAntenna = SerializableDescriptor(
        'TxAntenna', TxAntennaType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, TxAntennaType]

    def __init__(self, TxTime=None, TxPos=None, TxVel=None,
                 FX1=None, FX2=None, TXmt=None, TxLFM=None,
                 TxAntenna=None, **kwargs):
        """

        Parameters
        ----------
        TxTime : PerVectorParameterF8
        TxPos : PerVectorParameterXYZ
        TxVel : PerVectorParameterXYZ
        FX1 : PerVectorParameterF8
        FX2 : PerVectorParameterF8
        TXmt : PerVectorParameterF8
        TxLFM : None|PerVectorParameterTxLFM
        TxAntenna : None|TxAntennaType
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxTime = TxTime
        self.TxPos = TxPos
        self.TxVel = TxVel
        self.FX1 = FX1
        self.FX2 = FX2
        self.TXmt = TXmt
        self.TxLFM = TxLFM
        self.TxAntenna = TxAntenna
        super(TxPulseType, self).__init__(**kwargs)


class PVPType(Serializable):
    _fields = (
        'RcvTime', 'RcvPos', 'RcvVel', 'RefPhi0', 'RefFreq', 'DFIC0',
        'FICRate', 'FRCV1', 'FRCV2', 'DGRGC', 'SIGNAL', 'AmpSF',
        'RcvAntenna', 'TxPulse', 'AddedPVP')
    _required = (
        'RcvTime', 'RcvPos', 'RcvVel', 'RefPhi0', 'RefFreq', 'DFIC0',
        'FICRate', 'FRCV1', 'FRCV2')
    _collections_tags = {'AddedPVP': {'array': False, 'child_tag': 'AddedPVP'}}
    # descriptors
    RcvTime = SerializableDescriptor(
        'RcvTime', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    RcvPos = SerializableDescriptor(
        'RcvPos', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RcvVel = SerializableDescriptor(
        'RcvVel', PerVectorParameterXYZ, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterXYZ
    RefPhi0 = SerializableDescriptor(
        'RefPhi0', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    RefFreq = SerializableDescriptor(
        'RefFreq', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    DFIC0 = SerializableDescriptor(
        'DFIC0', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FICRate = SerializableDescriptor(
        'FICRate', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FRCV1 = SerializableDescriptor(
        'FRCV1', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    FRCV2 = SerializableDescriptor(
        'FRCV2', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PerVectorParameterF8
    DGRGC = SerializableDescriptor(
        'DGRGC', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, PerVectorParameterF8]
    SIGNAL = SerializableDescriptor(
        'SIGNAL', PerVectorParameterI8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, PerVectorParameterI8]
    AmpSF = SerializableDescriptor(
        'AmpSF', PerVectorParameterF8, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, PerVectorParameterF8]
    RcvAntenna = SerializableDescriptor(
        'RcvAntenna', RcvAntennaType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, RcvAntennaType]
    TxPulse = SerializableDescriptor(
        'TxPulse', TxPulseType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, TxPulseType]
    AddedPVP = SerializableListDescriptor(
        'AddedPVP', UserDefinedPVPType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, List[UserDefinedPVPType]]

    def __init__(self, RcvTime=None, RcvPos=None, RcvVel=None,
                 RefPhi0=None, RefFreq=None, DFIC0=None,
                 FICRate=None, FRCV1=None, FRCV2=None,
                 DGRGC=None, SIGNAL=None, AmpSF=None,
                 RcvAntenna=None, TxPulse=None,
                 AddedPVP=None, **kwargs):
        """

        Parameters
        ----------
        RcvTime : PerVectorParameterF8
        RcvPos : PerVectorParameterXYZ
        RcvVel : PerVectorParameterXYZ
        RefPhi0 : PerVectorParameterF8
        RefFreq : PerVectorParameterF8
        DFIC0 : PerVectorParameterF8
        FICRate : PerVectorParameterF8
        FRCV1 : PerVectorParameterF8
        FRCV2 : PerVectorParameterF8
        DGRGC : None|PerVectorParameterF8
        SIGNAL : None|PerVectorParameterI8
        AmpSF : None|PerVectorParameterF8
        RcvAntenna : None|RcvAntennaType
        TxPulse : None|TxPulseType
        AddedPVP : None|List[UserDefinedPVPType]
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RcvTime = RcvTime
        self.RcvPos = RcvPos
        self.RcvVel = RcvVel
        self.RefPhi0 = RefPhi0
        self.RefFreq = RefFreq
        self.DFIC0 = DFIC0
        self.FICRate = FICRate
        self.FRCV1 = FRCV1
        self.FRCV2 = FRCV2
        self.DGRGC = DGRGC
        self.SIGNAL = SIGNAL
        self.AmpSF = AmpSF
        self.RcvAntenna = RcvAntenna
        self.TxPulse = TxPulse
        self.AddedPVP = AddedPVP
        super(PVPType, self).__init__(**kwargs)

    def get_size(self):
        """
        Gets the size in bytes of each vector.

        Returns
        -------
        int
        """

        def get_num_words(obj):
            sz = getattr(obj, 'Size')
            if sz is not None:
                return sz
            sz = 0

            # noinspection PyProtectedMember
            for fld in obj._fields:
                fld_val = getattr(obj, fld)
                if fld_val is not None:
                    if fld_val.array:
                        for arr_val in fld_val:
                            sz += get_num_words(arr_val)
                    else:
                        sz += get_num_words(fld_val)
            return sz

        return get_num_words(self) * BYTES_PER_WORD

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

        def osf_tuple(val_in):
            return val_in.Offset*BYTES_PER_WORD, val_in.Size*BYTES_PER_WORD, homogeneous_dtype(val_in.Format).char

        # noinspection PyProtectedMember
        if field in self._fields[:-1]:
            val = getattr(self, field)
            if val is None:
                return None
            return osf_tuple(val)
        elif self.RcvAntenna and field in self.RcvAntenna._fields:
            val = getattr(self.RcvAntenna, field)
            if val is None:
                return None
            return osf_tuple(val)
        elif self.TxPulse and field in self.TxPulse._fields:
            val = getattr(self.TxPulse, field)
            if val is None:
                return None
            return osf_tuple(val)
        elif self.TxPulse and self.TxPulse.TxAntenna and field in self.TxPulse.TxAntenna._fields:
            val = getattr(self.TxPulse.TxAntenna, field)
            if val is None:
                return None
            return osf_tuple(val)
        else:
            if self.AddedPVP is None:
                return None
            for val in self.AddedPVP:
                if field == val.Name:
                    return osf_tuple(val)
        return None

    def get_vector_dtype(self):
        """
        Gets the dtype for the corresponding structured array for the full PVP set.

        Returns
        -------
        numpy.dtype
            This will be a compound dtype for a structured array.
        """

        names = []
        formats = []
        offsets = []

        for field in self._fields:
            val = getattr(self, field)
            if val is None:
                continue
            elif field == "AddedPVP":
                for entry in val:
                    names.append(entry.Name)
                    formats.append(binary_format_string_to_dtype(entry.Format))
                    offsets.append(entry.Offset*BYTES_PER_WORD)
            elif field == 'RcvAntenna' or field == 'TxPulse':
                continue
            else:
                names.append(field)
                formats.append(binary_format_string_to_dtype(val.Format))
                offsets.append(val.Offset*BYTES_PER_WORD)

        if self.RcvAntenna is not None:
            # noinspection PyProtectedMember
            for field in self.RcvAntenna._fields:
                val = getattr(self.RcvAntenna, field)
                if val is None:
                    continue
                else:
                    names.append(field)
                    formats.append(binary_format_string_to_dtype(val.Format))
                    offsets.append(val.Offset*BYTES_PER_WORD)

        if self.TxPulse is not None:
            # noinspection PyProtectedMember
            for field in self.TxPulse._fields:
                val = getattr(self.TxPulse, field)
                if val is None:
                    continue
                elif field == 'TxAntenna':
                    continue
                else:
                    names.append(field)
                    formats.append(binary_format_string_to_dtype(val.Format))
                    offsets.append(val.Offset*BYTES_PER_WORD)
            if self.TxPulse.TxAntenna is not None:
                # noinspection PyProtectedMember
                for field in self.TxPulse.TxAntenna._fields:
                    val = getattr(self.TxPulse.TxAntenna, field)
                    if val is None:
                        continue
                    else:
                        names.append(field)
                        formats.append(binary_format_string_to_dtype(val.Format))
                        offsets.append(val.Offset*BYTES_PER_WORD)

        return numpy.dtype({'names': names, 'formats': formats, 'offsets': offsets})
