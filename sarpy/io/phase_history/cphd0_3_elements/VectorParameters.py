# -*- coding: utf-8 -*-
"""
The SRP definition for CPHD 0.3.
"""

from typing import Union

from sarpy.compliance import integer_types
from sarpy.io.phase_history.cphd1_elements.base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _SerializableDescriptor, \
    _IntegerEnumDescriptor


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class FxParametersType(Serializable):
    """
    The FX vector parameters.
    """

    _fields = ('Fx0', 'Fx_SS', 'Fx1', 'Fx2')
    _required = _fields
    # descriptors
    Fx0 = _IntegerEnumDescriptor(
        'Fx0', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the Fx0 field')  # type: int
    Fx_SS = _IntegerEnumDescriptor(
        'Fx_SS', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the Fx_SS field')  # type: int
    Fx1 = _IntegerEnumDescriptor(
        'Fx1', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the Fx1 field')  # type: int
    Fx2 = _IntegerEnumDescriptor(
        'Fx2', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the Fx2 field')  # type: int

    def __init__(self, Fx0=8, Fx_SS=8, Fx1=8, Fx2=8, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Fx0 = Fx0
        self.Fx1 = Fx1
        self.Fx2 = Fx2
        self.Fx_SS = Fx_SS
        super(FxParametersType, self).__init__(**kwargs)

    @staticmethod
    def get_size():
        """
        The size in bytes of this component of the vector.

        Returns
        -------
        int
        """

        return 32

    def get_position_offset_and_size(self, field):
        """
        Get the offset and size of the given field from the beginning of the vector.

        Parameters
        ----------
        field : str

        Returns
        -------
        None|int
        """

        if field not in self._fields:
            return None

        out = 0
        for fld in self._fields:
            val = getattr(self, fld)
            if fld == field:
                return out, val
            else:
                out += val
        return None


class TOAParametersType(Serializable):
    """
    The TOA vector parameters.
    """

    _fields = ('DeltaTOA0', 'TOA_SS')
    _required = _fields
    # descriptors
    DeltaTOA0 = _IntegerEnumDescriptor(
        'DeltaTOA0', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the DeltaTOA0 field')  # type: int
    TOA_SS = _IntegerEnumDescriptor(
        'TOA_SS', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the TOA_SS field')  # type: int

    def __init__(self, DeltaTOA0=8, TOA_SS=8, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DeltaTOA0 = DeltaTOA0
        self.TOA_SS = TOA_SS
        super(TOAParametersType, self).__init__(**kwargs)

    @staticmethod
    def get_size():
        """
        The size in bytes of this component of the vector.

        Returns
        -------
        int
        """

        return 16

    def get_position_offset_and_size(self, field):
        """
        Get the offset and size of the given field from the beginning of the vector.

        Parameters
        ----------
        field : str

        Returns
        -------
        None|(int, int)
        """

        if field not in self._fields:
            return None

        out = 0
        for fld in self._fields:
            val = getattr(self, fld)
            if fld == field:
                return out, val
            else:
                out += val
        return None


class VectorParametersType(Serializable):
    """
    The vector parameters sizes object.
    """

    _fields = (
        'TxTime', 'TxPos', 'RcvTime', 'RcvPos', 'SRPTime', 'SRPPos', 'AmpSF', 'TropoSRP',
        'FxParameters', 'TOAParameters')
    _required = (
        'TxTime', 'TxPos', 'RcvTime', 'RcvPos', 'SRPPos')
    _choice = ({'required': False, 'collection': ('FxParameters', 'TOAParameters')}, )
    # descriptors
    TxTime = _IntegerEnumDescriptor(
        'TxTime', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the TxTime field')  # type: int
    TxPos = _IntegerEnumDescriptor(
        'TxPos', (24, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the TxPos field')  # type: int
    RcvTime = _IntegerEnumDescriptor(
        'RcvTime', (8, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the RcvTime field')  # type: int
    RcvPos = _IntegerEnumDescriptor(
        'RcvPos', (24, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the RcvPos field')  # type: int
    SRPTime = _IntegerEnumDescriptor(
        'SRPTime', (8, ), _required, strict=DEFAULT_STRICT, default_value=None,
        docstring='The size of the SRPTime field')  # type: int
    SRPPos = _IntegerEnumDescriptor(
        'SRPPos', (24, ), _required, strict=DEFAULT_STRICT, default_value=8,
        docstring='The size of the SRPPos field')  # type: int
    AmpSF = _IntegerEnumDescriptor(
        'AmpSF', (8, ), _required, strict=DEFAULT_STRICT, default_value=None,
        docstring='The size of the AmpSF field')  # type: int
    TropoSRP = _IntegerEnumDescriptor(
        'TropoSRP', (8, ), _required, strict=DEFAULT_STRICT, default_value=None,
        docstring='The size of the TropoSRP field')  # type: int
    FxParameters = _SerializableDescriptor(
        'FxParameters', FxParametersType, _required, strict=DEFAULT_STRICT,
        docstring='The frequency parameters, only present when DomainType is '
                  'FX.')  # type: Union[None, FxParametersType]
    TOAParameters = _SerializableDescriptor(
        'TOAParameters', FxParametersType, _required, strict=DEFAULT_STRICT,
        docstring='The TOA parameters, only present when DomainType is '
                  'TOA.')  # type: Union[None, TOAParametersType]

    def __init__(self, TxTime=8, TxPos=24, RcvTime=8, RcvPos=24, SRPTime=None, SRPPos=24,
                 AmpSF=None, TropoSRP=None, FxParameters=None, TOAParameters=None, **kwargs):
        """

        Parameters
        ----------
        TxTime : int
        TxPos : int
        RcvTime : int
        RcvPos : int
        SRPTime : None|int
        SRPPos : int
        AmpSF : None|int
        TropoSRP : None|int
        FxParameters : None|FxParametersType
        TOAParameters : None|TOAParametersType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxTime = TxTime
        self.TxPos = TxPos
        self.RcvTime = RcvTime
        self.RcvPos = RcvPos
        self.SRPTime = SRPTime
        self.SRPPos = SRPPos
        self.AmpSF = AmpSF
        self.TropoSRP = TropoSRP
        self.FxParameters = FxParameters
        self.TOAParameters = TOAParameters
        super(VectorParametersType, self).__init__(**kwargs)

    def get_size(self):
        """
        The size in bytes of this component of the vector.

        Returns
        -------
        int
        """

        out = 0
        for fld in self._fields:
            val = getattr(self, fld)
            if val is None:
                pass
            elif isinstance(val, integer_types):
                out += val
            elif isinstance(val, (FxParametersType, TOAParametersType)):
                out += val.get_size()
            else:
                raise TypeError('Got unhandled type {}'.format(type(val)))
        return out

    def get_position_offset_and_size(self, field):
        """
        Get the offset and size of the given field from the beginning of the vector.

        Parameters
        ----------
        field : str

        Returns
        -------
        None|(int, int)
        """

        out = 0
        for fld in self._fields:
            val = getattr(self, fld)
            if fld == field:
                if val is not None:
                    return out, val
                else:
                    return None

            if val is None:
                pass
            elif isinstance(val, integer_types):
                out += val
            elif isinstance(val, (FxParametersType, TOAParametersType)):
                res = val.get_position_offset_and_size(field)
                if res is not None:
                    return out+res[0], res[1]
                else:
                    out += val.get_size()
            else:
                raise TypeError('Got unhandled type {}'.format(type(val)))
        return None
